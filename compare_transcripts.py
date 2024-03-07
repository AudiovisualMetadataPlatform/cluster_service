#!/usr/bin/env python3

import activate_venv

import argparse
import jiwer
from pathlib import Path
import logging
import re
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true", help="Turn on debugging")
    parser.add_argument("--nopunc", default=False, action="store_true", help="Remove punctuation")
    parser.add_argument("--nocase", default=False, action="store_true", help="Case-insensitive")
    parser.add_argument("--differences", default=False, action="store_true", help="Only show edits where there's a difference")
    parser.add_argument('base', type=Path, help='Base whisper')
    parser.add_argument('comp', type=Path, help="Thing to compare to base")
    parser.add_argument('output', type=Path, help='Output file or directory')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s [%(process)d:%(filename)s:%(lineno)d] [%(levelname)s] %(message)s")
    
    todo = []
    if args.base.is_file():
        if not args.comp.is_file():
            logging.error("If base is a file then comp must also be a file")
            exit(1)
        logging.debug(f"Adding {args.base}, {args.comp}")
        todo.append([args.base, args.comp])
    elif args.base.is_dir():
        if not args.comp.is_dir():
            logging.error("If base is a directory then comp must also be a directory")
            exit(1)
        for f in args.base.glob("**/*.json"):            
            rf = f.relative_to(args.base)
            cf = (args.comp / rf)
            if not cf.exists():                
                logging.warning(f"Skipping base file {rf} as there isn't a corresponding file in {args.comp}")
                continue
            todo.append([f, cf])
    else:
        print(f"{args.base} is neither file nor dir? {args.base.stat()}")

    if not args.output.is_dir():
        is_file = True
        outfile = open(args.output, "w")
    else:
        is_file = False

    for b, c in todo:
        bdata = load_transcript(b)
        cdata = load_transcript(c)
        
        if args.nopunc:
            # spaceless punctuation
            bdata['text'] = re.sub(r"[_]+", '', bdata['text'])
            cdata['text'] = re.sub(r"[_]+", '', cdata['text'])

            # spaceful punctuation
            bdata['text'] = re.sub(r"[\-!@#$%^&*()+=\[\]{}\\|;:\",./<>?]+", ' ', bdata['text'])
            cdata['text'] = re.sub(r"[\-!@#$%^&*()+=\[\]{}\\|;:\",./<>?]+", ' ', cdata['text'])
            
        if args.nocase:
            bdata['text'] = bdata['text'].lower()
            cdata['text'] = cdata['text'].lower()
            
        o = jiwer.process_words(bdata['text'], cdata['text'])
        v, stats = generate_visualization(o, differences=args.differences)

        report = []
        report.extend(render_xscript_info("Base File", b, bdata))
        report.extend(render_xscript_info("Comp File", c, cdata))
        report.append("")

        report.extend([
            "Stats: ",
            f"  Word Error Rate:            {o.wer * 100:7.2f}%",
            f"  Word Information Lost:      {o.wil * 100:7.2f}%",
            f"  Word Information Preserved: {o.wip * 100:7.2f}%",
            f"  Match Error Rate:           {o.mer * 100:7.2f}%",
            "",
            "Edit Stats:",
            f"  Hits:          {stats['hit']:>5d}",
            f"  Inserts:       {stats['ins']:>5d}",
            f"  Deletes:       {stats['del']:>5d}",
            f"  Substitutions: {stats['sub']:>5d}",
            "",
            "Edits:"])
        for s in v:            
            report.append(f"BASE: {s['ref']}")
            report.append(f"COMP: {s['hyp']}")
            report.append(f"EDIT: {s['chg']}")
            report.append("")
        
        if is_file:
            report.append("-----------")
            outfile.write("\n".join(report) + "\n")
        else:
            with open(args.output / b.name, "w") as outfile:
                outfile.write("\n".join(report))

    if is_file:
        outfile.close()


def load_transcript(file):
    """Load a transcript file and convert it to the data structure we need for
    processing.  We'll support plain text, and yaml/json versions of whisper
    and AWS Transcribe"""
    try:
        with open(file) as f:
            raw = yaml.safe_load(f)
        
        xscript = {'media_duration': None,
                    'processing_time': None,
                    'engine': None,
                    'model': None,
                    'device': None,
                    'prompt': None,
                    'text': None}
        
        # look for formats...
        if type(raw) == str:
            xscript['engine'] = 'unknown'
            xscript['text'] = Path(file).read_text()        
        elif all([x in raw for x in ['jobName', 'accountId', 'results', 'status']]):
            # transcribe
            xscript['text'] = raw['results']['transcripts'][0]['transcript']
            xscript['engine'] = 'aws transcribe'            
        elif all(x in raw for x in ['text', 'segments', 'language']):
            # whisper
            xscript['text'] = raw['text']
            xscript['engine'] = 'whisper'
            # we may have job information...
            if '_job' in raw:
                xscript['processing_time'] = raw['_job'].get('runtime', None)
                xscript['model'] = raw['_job'].get('model', None)
                xscript['device'] = raw['_job'].get('device', None)
                xscript['prompt'] = raw['_job'].get('prompt', None)
        else:
            raise Exception(f"WTF?  Don't know this format: {raw}")

    except yaml.YAMLError:
        # not json or yaml, so assume it's text.
        xscript['engine'] = 'unknown'
        xscript['text'] = Path(file).read_text()           

    # normalize the text a bit:  get rid of newlines, carriage returns, and tabs    
    for n in ('\n', '\r', '\t'):
        xscript['text'] = xscript['text'].replace(n, " ")
    # and remove leading/trailing spaces.
    xscript['text'] = xscript['text'].strip()

    return xscript


def render_xscript_info(title: str, file: Path, data: dict):
    """Take the file and information and render it for the report"""
    report = []
    report.extend([
        f"{title}: {file.resolve()!s}"
    ])
    for n in (('media_duration', 'Media Duration'),
              ('processing_time', 'Processing Time')):
        if data.get(n[0], None):
            report.append(f"  {pad(n[1] + ':', 20)}{s2time(float(data[n[0]]))}")
    for n in (('engine', 'Engine'), 
              ('model', 'Model'),
              ('device', 'Device'),
              ('prompt', 'Prompt')):
        if data.get(n[0], None):
            report.append(f"  {pad(n[1] + ':', 20)}{data[n[0]]}")
    return report


def pad(word, pad_len):
    while len(word) < pad_len:
        word += " "
    return word


def generate_visualization(output: jiwer.WordOutput, length=75, differences=False):
    results = [{'ref': '', 'hyp': '', 'chg': '', 'dif': 0}]
    stats = {'hit': 0, 'sub': 0, 'del': 0, 'ins': 0}
    for idx, (gt, hp, chunks) in enumerate(zip(output.references, output.hypotheses, output.alignments)):
        #print(idx, gt, hp, chunks)
        for chunk in chunks:
            if chunk.type == 'equal':
                # copy ref, and hyp words until either we
                # end up too long or we come to the end.                    
                for i in range(chunk.ref_end_idx - chunk.ref_start_idx):                
                    stats['hit'] += 1
                    word_len = len(gt[i + chunk.ref_start_idx]) 
                    if word_len + len(results[-1]['ref']) + 1> length:
                        # too long. create a new result
                        results.append({'ref': '', 'hyp': '', 'chg': '', 'dif': 0})
            
                    results[-1]['ref'] += gt[i + chunk.ref_start_idx] + " "
                    results[-1]['hyp'] += hp[i + chunk.hyp_start_idx] + " "
                    results[-1]['chg'] += ' ' * (word_len + 1)     
            elif chunk.type == 'insert':
                # hyp has an additional word that's not in ref.                
                for i in range(chunk.hyp_end_idx - chunk.hyp_start_idx):                
                    stats['ins'] += 1
                    word_len = len(hp[i + chunk.hyp_start_idx])
                    if word_len + len(results[-1]['ref']) + 1> length:
                        # too long. create a new result
                        results.append({'ref': '', 'hyp': '', 'chg': '', 'dif': 0})
            
                    results[-1]['ref'] += ('*' * word_len) + " "
                    results[-1]['hyp'] += hp[i + chunk.hyp_start_idx] + " "
                    results[-1]['chg'] += ('I' * word_len) + " "
                    results[-1]['dif'] += 1
            elif chunk.type == 'delete':
                # ref has an additional word that's not in hyp.                
                for i in range(chunk.ref_end_idx - chunk.ref_start_idx): 
                    stats['del'] += 1               
                    word_len = len(gt[i + chunk.ref_start_idx])
                    if word_len + len(results[-1]['ref']) + 1> length:
                        # too long. create a new result
                        results.append({'ref': '', 'hyp': '', 'chg': '', 'dif': 0})
            
                    results[-1]['ref'] += gt[i + chunk.ref_start_idx] + " "
                    results[-1]['hyp'] += ('*' * word_len) + " "                    
                    results[-1]['chg'] += ('D' * word_len) + " "
                    results[-1]['dif'] += 1
            elif chunk.type == 'substitute':
                # ref and hyp have different words (but the same number)                
                for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    stats['sub'] += 1                
                    word_len = max([len(gt[i + chunk.ref_start_idx]),
                                    len(hp[i + chunk.hyp_start_idx])])
                    if word_len + len(results[-1]['ref']) + 1> length:
                        # too long. create a new result
                        results.append({'ref': '', 'hyp': '', 'chg': '', 'dif': 0})
            
                    results[-1]['ref'] += pad(gt[i + chunk.ref_start_idx], word_len) + " "
                    results[-1]['hyp'] += pad(hp[i + chunk.hyp_start_idx], word_len) + " "
                    results[-1]['chg'] += 'S' * (word_len) + ' '
                    results[-1]['dif'] += 1
            else:
                print(chunk)

    if differences:
        return [x for x in results if x['dif'] > 0], stats
    else:
        return results, stats


def s2time(seconds):        
    hours = int(seconds / 3600)
    seconds -= hours * 3600
    minutes = int(seconds / 60)
    seconds -= minutes * 60
    return f"{hours:0d}:{minutes:02d}:{seconds:06.3f}"


if __name__ == "__main__":
    main()