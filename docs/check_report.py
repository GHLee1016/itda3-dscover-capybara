"""본선보고서 PDF 가 규정을 지키는가."""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import pymupdf

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "[DScover]_카피바라_본선보고서.pdf")
d = pymupdf.open(P)
w_mm = d[0].rect.width / 72 * 25.4
h_mm = d[0].rect.height / 72 * 25.4

print("=" * 60)
print(" 규정 점검 — [DScover]_카피바라_본선보고서.pdf")
print("=" * 60)


def chk(name, ok, detail=""):
    print("  %s %-34s %s" % ("OK " if ok else "!! ", name, detail))
    return ok


res = []
res.append(chk("PDF 형식", True, "%d KB" % (os.path.getsize(P) / 1024)))
res.append(chk("A4 용지", abs(w_mm - 210) < 2 and abs(h_mm - 297) < 2,
               "%.0f x %.0f mm" % (w_mm, h_mm)))
body = d.page_count - 2          # 표지 1 + 목차 1 은 분량 제외
res.append(chk("분량 (표지·목차 제외 10쪽 이하)", body <= 10,
               "전체 %d쪽 → 본문 %d쪽" % (d.page_count, body)))

txt = "\n".join(p.get_text() for p in d)
res.append(chk("텍스트 추출 가능 (스캔본 아님)", len(txt) > 8000, "%d자" % len(txt)))
res.append(chk("첫 목차가 '1. 개괄'", "1. 개괄" in txt))
res.append(chk("팀명 기재", "[DScover]_카피바라" in txt))
res.append(chk("8개 장 모두 존재",
               all(("%d. " % i) in txt for i in range(1, 9))))

# 필수 수치
for k in ("92.3%", "1.20초", "614초", "26%", "0건", "828"):
    res.append(chk("수치 %s" % k, k in txt))

# 양식의 필수 소절
need = ["1-1", "1-2", "2-1", "2-2", "2-3", "3-1", "3-2", "3-3",
        "4-1", "4-2", "4-3", "5-1", "5-2", "5-3", "6-1", "6-2", "6-3", "6-4"]
miss = [n for n in need if n not in txt]
res.append(chk("양식 소절 18개 전부", not miss, "빠짐: %s" % miss if miss else ""))

res.append(chk("자리표시자 없음",
               "<ORG>" not in txt and "TODO" not in txt and "팀명_" not in txt))
res.append(chk("GPU 비용 명시 (규정 필수)",
               "0원" in txt and "GPU" in txt))

print()
print(" 쪽별 글자 수 (빈 쪽 확인)")
for i, p in enumerate(d, 1):
    n = len(p.get_text().strip())
    tag = "  표지" if i == 1 else ("  목차" if i == 2 else "")
    flag = "  ← 거의 빈 쪽" if n < 120 and i > 2 else ""
    print("   %2d쪽  %5d자%s%s" % (i, n, tag, flag))

print()
print(" 통과 %d / %d" % (sum(res), len(res)))
sys.exit(0 if all(res) else 1)
