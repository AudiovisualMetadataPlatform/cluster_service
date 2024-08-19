#!/usr/bin/env python3

import activate_venv
from cservice import ClusterService
import logging
import json
import torch
import whisper
import time


class Whisper(ClusterService):

    def filter_jobdata(self, jobdata: dict):
        # update the manifest.  If there's a .whisper.json file for a manifest
        # entry, we'll remove the entry.  If there's nothing left in the manifest
        # then we won't add it to the jobs since it's effectively finished.
        for file in list(jobdata['manifest']):
            if file and (jobdata['jobdir'] / (file + ".whisper.json")).exists():
                jobdata['manifest'].remove(file)


    def work(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Whisper will use {device} for computation")
        while True:
            todo = self.get_todo_list("whisper.job")
            if not todo:
                break
            # if none of the items in todo are "ready" then we're going to sleep a bit
            logging.info(f"Starting processing of {len(todo)} jobs")

            # get the models we need to use...
            process_count = 0
            models = {x['model'] for x in todo}
            for model_name in models:
                logging.info(f"Starting processing for jobs using {model_name} model")
                model = whisper.load_model(model_name, device,)
                for t in todo:
                    if t['model'] != model_name:
                        continue
                    logging.info(f"Processing job in {t['jobdir']}")
                    for f in t['manifest']:
                        try:
                            logging.info(f"Starting processing of {f}")
                            audio = whisper.load_audio(t['jobdir'] / f)

                            if 'prompt' not in t:
                                t['prompt'] = None

                            if 'language' not in t or t['language'] == 'auto':
                                probable_languages = ('en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja')                
                                detect_audio = whisper.pad_or_trim(audio)
                                # This is a bug https://github.com/openai/whisper/discussions/1778
                                # basically, the v3 large model uses 128 mels, and the others use 80.
                                mel = whisper.log_mel_spectrogram(detect_audio, n_mels=128 if model_name in ('large', 'large-v3') else 80).to(device)
                                _, probs = model.detect_language(mel)            
                                probs = {k: v for k, v in probs.items() if k in probable_languages}                        
                                language = max(probs, key=probs.get)                          
                                logging.info(f"Language detection: {language}:  {probs}")
                            else:
                                language = t['language']

                            start = time.time()                    
                            res = whisper.transcribe(model, audio, word_timestamps=True, language=language,
                                                    verbose=None, initial_prompt=t['prompt'])
                            duration = time.time() - start

                            res['_job'] = {
                                'runtime': duration,
                                'device': device,
                                'language': language,
                                'model': t['model'],
                                'prompt': t['prompt']
                            }

                            with open(t['jobdir'] / (f + ".whisper.json"), "w") as f:
                                json.dump(res, f, indent=4)
                            process_count += 1

                        except Exception as e:
                            logging.exception(f"Failed during transcription of {f}")

                # unload the model to the best of our ability
                del(model)
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
            if process_count == 0:
                logging.warning("Some processes didn't complete even though they were valid")
                break


if __name__ == "__main__":
    Whisper().main()
