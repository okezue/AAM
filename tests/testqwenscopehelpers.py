from __future__ import annotations
from aamemory.encoding.qwenscope import infertokentopk
def testqwenscopetopkisinferredfromrepositoryname() -> None:
    assert infertokentopk("Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50") == 50
    assert infertokentopk("custom/repo", default=17) == 17
