"""
Separate genuine phone-captured images from stock/web downloads.
WhatsApp images are the only ones guaranteed to be real phone captures.
Run this before train.py.
"""
import os, shutil, argparse

def clean(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    kept, removed = [], []
    for f in sorted(os.listdir(src_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext not in {'.jpg','.jpeg','.png','.webp','.avif'}:
            continue
        is_genuine = f.lower().startswith('whatsapp')
        (kept if is_genuine else removed).append(f)
        if is_genuine:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    print(f"{src_dir}: kept {len(kept)}, removed {len(removed)}")
    for r in removed:
        print(f"  excluded: {r}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_in",   default="../dataset/real")
    ap.add_argument("--screen_in", default="../dataset/screen")
    ap.add_argument("--real_out",   default="../dataset/real_clean")
    ap.add_argument("--screen_out", default="../dataset/screen_clean")
    args = ap.parse_args()
    clean(args.real_in,   args.real_out)
    clean(args.screen_in, args.screen_out)
