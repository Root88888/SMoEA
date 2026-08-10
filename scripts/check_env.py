#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/check_env.py

【環境一鍵體檢】任何工作開始前先跑本腳本；全部 PASS 才代表環境正確。
檢查三類問題（全部是實際踩過的雷）：
  1. 版本：核心套件是否等於鎖定檔基準（requirements-lock-twcc.txt）；
  2. 來源：每個套件的載入路徑是否來自「當前環境」——抓 user-site
     （~/.local）與 conda 疊層造成的冒牌套件；
  3. 防護：PYTHONNOUSERSITE 是否啟用（阻斷 user-site 汙染的根治開關）。

【執行】python scripts/check_env.py
結束碼 0=全 PASS；非 0=列出每條 FAIL 與處置提示。
非國網環境版本不同屬正常，看第 2、3 類是否乾淨即可。
"""

import os
import sys

EXPECT = {          # 國網基準（requirements-lock-twcc.txt）
    "numpy": "1.26.4", "scipy": "1.17.1", "sklearn": "1.8.0",
    "transformers": "4.46.3", "peft": "0.14.0",
    "torch": "2.4.1",           # 前綴比對（+cu121 後綴）
}

fails = []


def check(name, ok, detail, hint=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        fails.append((name, hint))


print("== SMoEA 環境體檢 ==")
print(f"python {sys.version.split()[0]} @ {sys.executable}")

env_prefix = sys.prefix
check("python 來源", "/envs/" in env_prefix or "conda" in env_prefix.lower()
      or os.environ.get("CONDA_PREFIX", "") in (env_prefix,) and env_prefix,
      env_prefix, "未在 conda env 內——先 source env.sh")

check("PYTHONNOUSERSITE", os.environ.get("PYTHONNOUSERSITE") == "1",
      os.environ.get("PYTHONNOUSERSITE", "(未設)"),
      "export PYTHONNOUSERSITE=1（env.sh 應包含；沒有它 ~/.local 殘留會滲入）")

stack = int(os.environ.get("CONDA_SHLVL", "0") or 0)
check("conda 疊層", stack <= 1, f"CONDA_SHLVL={stack}",
      "疊層>1：while [ -n \"$CONDA_DEFAULT_ENV\" ]; do conda deactivate; done 後重進")

for mod, want in EXPECT.items():
    try:
        m = __import__(mod)
        ver = m.__version__
        path = getattr(m, "__file__", "") or ""
        v_ok = ver.startswith(want)
        p_ok = path.startswith(os.path.join(env_prefix, "lib"))
        check(f"{mod} 版本", v_ok, f"{ver}（基準 {want}）",
              "pip install -r requirements-lock-twcc.txt")
        check(f"{mod} 來源", p_ok, path,
              "套件來自環境外（user-site/其他 python）——見 FAIL 處置")
    except ImportError as e:
        check(f"{mod}", False, f"import 失敗：{e}",
              "pip install -r requirements-lock-twcc.txt")

try:
    import torch
    check("CUDA", torch.cuda.is_available(),
          f"available={torch.cuda.is_available()}",
          "無卡容器屬正常（selftest 不需 GPU）；生成/裁決需 GPU 容器")
except ImportError:
    pass

bad_local = [d for d in ("~/.local/lib/python3.10", "~/.local/lib/python3.12",
                         "~/.local/lib/python3.13")
             if os.path.isdir(os.path.expanduser(d))]
protected = os.environ.get("PYTHONNOUSERSITE") == "1"
check("user-site 隔離", protected or not bad_local,
      ("無殘留" if not bad_local else
       f"{', '.join(bad_local)}（已由 PYTHONNOUSERSITE=1 隔離，無害）"
       if protected else ", ".join(bad_local)),
      "設 PYTHONNOUSERSITE=1（setup_workspace.sh 會固化進 env），"
      "或 mv ~/.local/lib ~/.local/lib.graveyard")

print()
if fails:
    print(f"共 {len(fails)} 條 FAIL：")
    for name, hint in fails:
        print(f"  - {name}：{hint}")
    sys.exit(1)
print("環境體檢全部 PASS")
