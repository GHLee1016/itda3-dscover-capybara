"""편의점 소비기한 OCR 도입 ROI (보고서 6장). 가정만 바꿔 다시 돌리면 표가 갱신된다."""
# ---- 실측 (미사용 248장, 500장 순차) ----
P_EXACT = 0.895      # 완전일치
MEAN_S  = 1.01       # 장당 평균 초
P1, T1  = 0.80, 0.45 # 1칸 종료 비율, 1칸 평균 초(중앙값 0.39에서 보수적으로)
# ---- 출처 ----
STORES   = 53266     # 2025 편의점 점포 수 (포춘코리아 2026.9)
SALES    = 24.7e12   # 2025 편의점 매출 합계
WASTE_R  = 0.01      # 폐기 = 매출의 약 1% (헤럴드경제 2022.12, 점포당 월 30~80만)
WAGE     = 10320     # 2026 최저임금
WAGE_EFF = WAGE*1.2  # 주휴수당 반영 실질 시급
# ---- 가정 ----
ITEMS   = 300        # 점포당 일 입고·확인 건수
T_MAN   = 5.0        # 수작업 1건 초 (집기·판독·기록)
T_AUTO  = 2.0        # 촬영·확인 1건 초
ADDR    = 0.40       # 폐기 중 가공식품(바코드에 기한 없음) 비중
RECOVER = 0.20       # 조기 발견으로 할인판매 전환되는 비율
DAYS    = 30

esc = (MEAN_S - P1*T1)/(1-P1)
print(f"승급 이미지 평균 {esc:.2f}초")
monthly_sales = SALES/STORES/12
waste = monthly_sales*WASTE_R
print(f"점포당 월매출 {monthly_sales/1e4:,.0f}만, 월 폐기 {waste/1e4:.1f}만")

def per_store(items=ITEMS, t_man=T_MAN, addr=ADDR, rec=RECOVER):
    save_s = P_EXACT*(t_man-T_AUTO) - (1-P_EXACT)*T_AUTO
    labor = items*save_s*DAYS/3600*WAGE_EFF
    wst = waste*addr*rec
    return save_s, labor, wst
s, lab, wst = per_store()
print(f"건당 절감 {s:.2f}초, 인건비 월 {lab/1e4:.1f}만, 폐기 절감 월 {wst/1e4:.1f}만, 합 {(lab+wst)/1e4:.1f}만/월, 연 {(lab+wst)*12/1e4:.0f}만")

# ---- 체인 비용 (본부 1곳, N점) ----
DEV = 2*6*7_000_000          # 앱 이식·검증 2명×6개월
def chain(N, lab=lab, wst=wst):
    ops  = 2*80_000_000 if N>=2000 else 0.5*80_000_000
    field= N*20_000                     # 점포 교육·CS
    srv  = 12*3*147.17*1400             # 동기화·대시보드·모델배포 서버 3대 (추론은 단말)
    cost = DEV/3 + ops + field + srv
    ben  = N*(lab+wst)*12
    return ben, cost
for N in (100, 1000, 10000):
    b,c = chain(N)
    print(f"{N:>6}점: 편익 {b/1e8:6.2f}억, 비용 {c/1e8:5.2f}억, 순 {(b-c)/1e8:6.2f}억, ROI {(b-c)/c:5.1f}배, 개발비 회수 {DEV/((b-c+DEV/3)/12):.1f}개월")

print("\n민감도 (1,000점, ROI 배)  행=가공식품 비중, 열=회수율")
for a in (0.2,0.4,0.6):
    row=[]
    for r in (0.1,0.2,0.3):
        _,l,w = per_store(addr=a, rec=r)
        b,c = chain(1000,l,w); row.append(f"{(b-c)/c:5.1f}")
    print(a,row)
print("민감도 (1,000점, ROI 배)  행=일 건수, 열=수작업 초")
for it in (150,300,500):
    row=[]
    for tm in (3,5,8):
        _,l,w = per_store(items=it, t_man=tm)
        b,c = chain(1000,l,w); row.append(f"{(b-c)/c:5.1f}")
    print(it,row)

# ---- 온디바이스 부하 ----
print(f"\n점포 단말 부하: {ITEMS}건×{MEAN_S}s = {ITEMS*MEAN_S/60:.1f}분/일 (개발PC 기준), 모바일 3배 느리면 {ITEMS*MEAN_S*3/60:.0f}분 백그라운드")
# ---- 물류센터 서버 ----
IMG=100_000; HOURS=16
cpu_h = IMG*MEAN_S/3600
inst = -(-cpu_h//HOURS)
print(f"물류센터 일 {IMG:,}장: {cpu_h:.1f} 인스턴스-시간/일, {HOURS}시간 가동 → {inst:.0f}대 + 이중화 1대")
for n,label in ((2,'팀 초안 2대'),(inst+1,'권장')):
    print(f"  {label}: 온디맨드 월 {n*147.17*1400/1e4:.1f}만, 3년약정 월 {n*66.89*1400/1e4:.1f}만")
print(f"오탐 0/248 → 95% 상한 {3/248*100:.1f}%")
