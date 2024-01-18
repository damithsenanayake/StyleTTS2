import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import nltk
nltk.download('punkt')
from stts2ISN.Modules.diffusion import sampler
# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from stts2ISN.models import *
from stts2ISN.utils import *
from stts2ISN.text_utils import TextCleaner
import phonemizer
import os
from stts2ISN.Utils.PLBERT.util import load_plbert

import site

class StyleTTS2Synthesizer():

    def __init__(self, styletts2_model_path):
        self.textclenaer = TextCleaner()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean, self.std = -4, 4


        self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')

        config = yaml.safe_load(open(os.path.join(styletts2_model_path, "Models/LJSpeech/config.yml")))

        # load pretrained ASR model
        ASR_config = os.path.join(styletts2_model_path,config.get('ASR_config', False))
        print(ASR_config)
        ASR_path = os.path.join(styletts2_model_path,config.get('ASR_path', False))
        print(ASR_path)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = os.path.join(styletts2_model_path,config.get('F0_path', False))
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model
        BERT_path = os.path.join(styletts2_model_path, config.get('PLBERT_dir', False))
        plbert = load_plbert(BERT_path)

        self.model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        params_whole = torch.load(os.path.join(styletts2_model_path, "Models/LJSpeech/epoch_2nd_00100.pth"), map_location='cpu')
        params = params_whole['net']

        for key in self.model:
            if key in params:
                print('%s loaded' % key)
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], model[key])
        _ = [self.model[key].eval() for key in self.model]

        self.sampler=sampler.DiffusionSampler(
                    self.model.diffusion.diffusion,
                    sampler=sampler.ADPM2Sampler(),
                    sigma_schedule=sampler.KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
                    clamp=False
                )
        
        params_whole = torch.load(os.path.join(styletts2_model_path, "Models/LJSpeech/epoch_2nd_00100.pth"), map_location='cpu')
        self.params = params_whole['net']


    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self, ref_dicts):
        reference_embeddings = {}
        for key, path in ref_dicts.items():
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                audio = librosa.resample(audio, sr, 24000)
            mel_tensor = self.preprocess(audio).to(self.device)

            with torch.no_grad():
                ref = self.model.style_encoder(mel_tensor.unsqueeze(1))
            reference_embeddings[key] = (ref.squeeze(1), audio)

        return reference_embeddings
    
    

   
    def inference(self, text, noise, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        text = text.replace('"', '')
        ps = self.global_phonemizer.phonemize([text])
        print(f'phonemized text :{ps}')

        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise,
                embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                embedding_scale=embedding_scale).squeeze(0)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_dur[-1] += 5

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))
            print('output generated')
        return out.squeeze().cpu().numpy()

    def LFinference(self, text, s_prev, noise, alpha=0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        text = text.replace('"', '')
        ps = self.global_phonemizer.phonemize([text])
        print(f'phonemized text :{ps}')
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = sampler(noise,
                    embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                    embedding_scale=embedding_scale).squeeze(0)

            if s_prev is not None:
                # convex combination of previous and current style
                s_pred = alpha * s_prev + (1 - alpha) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy(), s_pred