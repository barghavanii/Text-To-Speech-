import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig , CharactersConfig
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig, VitsArgs
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager

#import wandb
 # Start a wandb run with `sync_tensorboard=True`
#if wandb.run is None:
    #wandb.init(project="persian-tts-vits-grapheme-cv15-fa-male-native-multispeaker-RERUN", group="GPUx8 accel mixed bf16 128x32", sync_tensorboard=True)

# output_path = os.path.dirname(os.path.abspath(__file__))
# output_path = output_path + '/notebook_files/runs'
# output_path = wandb.run.dir  ### PROBABLY better for notebook
output_path = "runs"

# print("output path is:")
# print(output_path)

cache_path = "cache"



# def mozilla(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
#     """Normalizes Mozilla meta data files to TTS format"""
#     txt_file = os.path.join(root_path, meta_file)
#     items = []
#     # speaker_name = "mozilla"
#     with open(txt_file, "r", encoding="utf-8") as ttf:
#         for line in ttf:
#             cols = line.split("|")
#             wav_file = cols[1].strip()
#             text = cols[0].strip()
#             speaker_name = cols[2].strip()
#             wav_file = os.path.join(root_path, "wavs", wav_file)
#             items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
#     return items



dataset_config = BaseDatasetConfig(
    formatter='common_voice', meta_file_train='validated.tsv', path="/home/bargh1/TTS/datasets"
)




character_config=CharactersConfig(
  characters='ءابتثجحخدذرزسشصضطظعغفقلمنهويِپچژکگیآأؤإئًَُّ',
#   characters="!¡'(),-.:;¿?ABCDEFGHIJKLMNOPRSTUVWXYZabcdefghijklmnopqrstuvwxyzáçèéêëìíîïñòóôöùúûü«°±µ»$%&‘’‚“`”„",
  punctuations='!(),-.:;? ̠،؛؟‌<>٫',
  phonemes='ˈˌːˑpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟaegiouwyɪʊ̩æɑɔəɚɛɝɨ̃ʉʌʍ0123456789"#$%*+/=ABCDEFGHIJKLMNOPRSTUVWXYZ[]^_{}۱۲۳۴۵۶۷۸۹۰',
  pad="<PAD>",
  eos="<EOS>",
  bos="<BOS>",
  blank="<BLNK>",
  characters_class="TTS.tts.models.vits.VitsCharacters",
  )

# From the coqui multilinguL recipes, will try later
vitsArgs = VitsArgs(
    # use_language_embedding=True,
    # embedded_language_dim=1,
    use_speaker_embedding=True,
    use_sdp=False,
)

audio_config = BaseAudioConfig(
     sample_rate=22050,
     do_trim_silence=True,
     min_level_db=-1,
    # do_sound_norm=True,
     signal_norm=True,
     clip_norm=True,
     symmetric_norm=True,
     max_norm = 0.9,
     resample=True,
     win_length=1024,
     hop_length=256,
     num_mels=80,
     mel_fmin=0,
     mel_fmax=None
 )

vits_audio_config = VitsAudioConfig(
    sample_rate=22050,
#    do_sound_norm=True,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    # do_trim_silence=True, #from hugging
    mel_fmin=0,
    mel_fmax=None
)
config = VitsConfig(
    model_args=vitsArgs,
    audio=vits_audio_config, #from huggingface
    run_name="persian-tts-vits-grapheme-cv15-multispeaker-RERUN",
    use_speaker_embedding=True, ## For MULTI SPEAKER
    batch_size=8,
    batch_group_size=16,
    eval_batch_size=4,
    num_loader_workers=16,
    num_eval_loader_workers=8,
    run_eval=True,
    run_eval_steps = 1000,
    print_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    save_step=1000,
    text_cleaner="basic_cleaners", #from MH
    use_phonemes=False,
    # phonemizer='persian_mh', #from TTS github
    # phoneme_language="fa",
    characters=character_config, #test without as well
    phoneme_cache_path=os.path.join(cache_path, "phoneme_cache_grapheme_azure-2"),
    compute_input_seq_cache=True,
    print_step=25,
    mixed_precision=False, #from TTS - True causes error "Expected reduction dim"
    test_sentences=[
        ["زین همرهان سست عناصر، دلم گرفت."],
        ["بیا تا گل برافشانیم و می در ساغر اندازیم."],
        ["بنی آدم اعضای یک پیکرند, که در آفرینش ز یک گوهرند."],
        ["سهام زندگی به 10 درصد و سهام بیتکوین گوگل به 33 درصد افزایش یافت."],
        ["من بودم و آبجی فوتینا، و حالا رپتی پتینا. این شعر یکی از اشعار معروف رو حوضی است که در کوچه بازار تهران زمزمه می شده است." ],
        ["یه دو دقه هم به حرفم گوش کن، نگو نگوشیدم و نحرفیدی."],
        [ "داستان با توصیف طوفان‌های شدید آغاز می‌شود؛ طوفان‌هایی که مزرعه‌ها را از بین می‌برد و محصولات را زیر شن دفن می‌کند؛ محصولاتی که زندگی افراد بسیاری به آن وابسته است."]
    ],
    output_path=output_path,
    datasets=[dataset_config]
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.num_speakers = speaker_manager.num_speakers



# init model
model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)

# init the trainer and 🚀

trainer = Trainer(
    TrainerArgs(use_accelerate=True),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
