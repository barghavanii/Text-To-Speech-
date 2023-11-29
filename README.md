# Text-To-Speech-
VITS and XTTS
Demo can be found in this link :https://huggingface.co/spaces/saillab/ZabanZad_PoC
# Prerequisities: 

1. Python >= 3.9
2. Espeak-NG : `sudo apt install -y espeak-ng`
3. TTS (from the repo):
   `pip install -U pip setuptools wheel`
   `git clone https://github.com/coqui-ai/TTS`
   `pip install -e TTS/`



# Setup Environment 
1.  init.sh--> add "sudo apt update && sudo apt upgrade -y

sudo apt install -y python3.10 python3.10-dev python3.10-venv
/usr/bin/python3.10 -m venv /opt/python/envs/py310
/opt/python/envs/py310/bin/python -m pip install -U pip setuptools wheel
/opt/python/envs/py310/bin/python -m pip install -U ipykernel ipython ipython_genutils jedi lets-plot aiohttp pandas

sudo apt install -y espeak-ng"


2.  in attached data --> on file environment.yml -->change datalore-base-env:"minimal" to  "py310"
3.  background computation --> Never Trun off 
4.  git clone https://github.com/coqui-ai/TTS.git
5.  navigate to TTS (cd TTS)
6.  pip install -e.



# Run Multi-GPU
1.  CUDA_VISIBLE_DEVICES="0,1" accelerate launch --multi_gpu --num_processes 2 multi-speaker.py
2.  For avoiding any intruption over your training use trainer = Trainer(
    TrainerArgs(use_accelerate=True),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
3.  Faster train with more than 1 num_loader_workers=4 in advance you should do sudo mount -o remount,size=8G /dev/shm



# Run with one GPU

1.  !nvidia-smi(status of GPU)
2.  os.environ["CUDA_VISIBLE_DEVICES"] = "7" which GPU you intend to run your code
# How to fix Error over runtime 
1.  Error :"torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacty of 9.77 GiB of which 52.31 MiB is free. Including non-PyTorch memory, this process has 8.68 GiB memory in use. Of the allocated memory 8.25 GiB is allocated by PyTorch, and 155.23 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF"
   Solution : reduce batch_size
2.  Error: Can't find some wavs although they are existed
    Solution : your wavs might be nested in files and cannot find nested files
3.  if you use common-voice as your formatter your wavs must be store in clips
4.  Error :dimension 
    Solution : mix precesion =false




# Tensorboard :
1.  For tensorboard download the latest output and un zip it go to that file run this command on window shell go cd to the address you stored the file : tensorboard --logdir=. --bind_all --port=6007 --> url open in your browser
2.  Use wandb -->import wandb
 Start a wandb run with `sync_tensorboard=True`
if wandb.run is None:
    wandb.init(project="persian-tts-vits-grapheme-cv15-fa-male-native-multispeaker-RERUN", group="GPUx8 accel mixed bf16 128x32", sync_tensorboard=True)



# For Multi-Speaker 
1.  use_speaker_embedding=True
2.  speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.num_speakers = speaker_manager.num_speakers
3.  model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)
4.  to run with multi-GPU : CUDA_VISIBLE_DEVICES="0,1" accelerate launch --multi_gpu --num_processes 2 multi-speaker.py

## Push log to hub 
1.  Navigate to stored log
2.  use python code titled "push_to_hub

# experiment 
run multi-speaker and then go to single speaker it removed the capability of model to be multi- speaker . 
for load your previous checkpoint navigate to the place that you want to resume your training and on that .py file add "model.load_checkpoint(config, 'best_model_495.pth', eval=False)" and load your modle from the checkpoint that you wish to restart 
I started an experiment dated 9 Nov cv--> azure_male-->azure_female and it seams that although it is becoming single speaker you don't need to change model configuration and it will keep running 


# XTTS V2
follow this link https://github.com/coqui-ai/TTS/issues/3229
but still got error 


