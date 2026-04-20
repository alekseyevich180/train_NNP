#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from pathlib import Path


def is_set_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("set")


def safe_remove(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def prepare_dir(path: Path, force: bool):
    if path.exists() or path.is_symlink():
        if not force:
            raise FileExistsError(f"{path} 已存在，请加 --force")
        safe_remove(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_mode: bool):
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"目标已存在: {dst}")
    if copy_mode:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def split_sets(set_dirs, val_ratio, seed):
    rng = random.Random(seed)
    s = list(set_dirs)
    rng.shuffle(s)
    n_total = len(s)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_val = min(n_val, n_total - 1)
    return s[n_val:], s[:n_val]  # train, val


def update_input_json(input_json: Path, train_systems, val_systems):
    data = json.loads(input_json.read_text(encoding="utf-8"))
    data["training"]["training_data"]["systems"] = train_systems
    data["training"]["validation_data"]["systems"] = val_systems
    input_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="将各体系中的 set* 统一整理到 root/train 和 root/val（按体系分目录）"
    )
    parser.add_argument("root", type=Path, help="例如 /home/.../nnp_train")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="默认 0.2（约4:1）")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--copy", action="store_true", help="复制文件（默认软链接）")
    parser.add_argument("--force", action="store_true", help="覆盖已有 train/val")
    parser.add_argument("--update-input", type=Path, default=None, help="可选：自动更新 input.json")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    out_train = root / "train"
    out_val = root / "val"
    prepare_dir(out_train, args.force)
    prepare_dir(out_val, args.force)

    system_dirs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in {"train", "val"}:
            continue
        if any(is_set_dir(x) for x in d.iterdir()):
            system_dirs.append(d)

    if not system_dirs:
        print("未找到包含 set* 的体系目录")
        return

    train_systems_out = []
    val_systems_out = []

    for i, sys_dir in enumerate(system_dirs):
        set_dirs = sorted([p for p in sys_dir.iterdir() if is_set_dir(p)])
        if len(set_dirs) < 2:
            print(f"[跳过] {sys_dir.name}: set 数量不足 2")
            continue

        train_sets, val_sets = split_sets(set_dirs, args.val_ratio, args.seed + i)

        dst_train_sys = out_train / sys_dir.name
        dst_val_sys = out_val / sys_dir.name
        dst_train_sys.mkdir(parents=True, exist_ok=False)
        dst_val_sys.mkdir(parents=True, exist_ok=False)

        # 公共文件（type.raw/type_map.raw/nopbc等）
        for item in sys_dir.iterdir():
            if is_set_dir(item):
                continue
            link_or_copy(item, dst_train_sys / item.name, args.copy)
            link_or_copy(item, dst_val_sys / item.name, args.copy)

        for s in train_sets:
            link_or_copy(s, dst_train_sys / s.name, args.copy)
        for s in val_sets:
            link_or_copy(s, dst_val_sys / s.name, args.copy)

        train_systems_out.append(str(dst_train_sys.resolve()))
        val_systems_out.append(str(dst_val_sys.resolve()))

        print(f"[完成] {sys_dir.name}: train={len(train_sets)}, val={len(val_sets)}")

    if args.update_input:
        update_input_json(args.update_input.resolve(), train_systems_out, val_systems_out)
        print(f"[完成] 已更新 {args.update_input}")


if __name__ == "__main__":
    main()