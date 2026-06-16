# FnGuide first-pass coverage

- ETFs covered: 129
- Methodology PDFs available: 126
- Methodology PDFs missing: 3

## Readiness

- needs_external_model_data: 79
- needs_dividend_or_custom_data: 29
- needs_core_market_data: 18
- blocked_missing_pdf: 3

## Families

- keyword_theme: 70
- dividend: 29
- sector_theme: 17
- equal_weight: 9
- unknown: 3
- top_n_theme: 1

## Missing Data

- constituent_universe: 129
- corporate_actions: 129
- krx_trading_calendar: 129
- listed_shares: 129
- stock_prices: 129
- free_float_ratio: 126
- market_cap: 125
- trading_value_liquidity: 113
- futures_options_expiry_calendar: 104
- ranking_inputs: 104
- fics_industry_classification: 85
- keyword_source_documents: 79
- dividend_data: 29
- custom_index_formula_inputs: 9
- score_inputs: 8
- fundamental_factors: 3
- methodology_pdf_missing: 3
- month_end_business_day_calendar: 2

## ETF Actions

| code | name | index | readiness | next action |
| --- | --- | --- | --- | --- |
| 0182R0 | 1Q K반도체TOP2+ | FnGuide AI Semiconductor TOP2+ Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0103T0 | 1Q K소버린AI | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 385600 | ACE 2차전지&친환경차액티브 | FnGuide Secondary Battery Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 469150 | ACE AI반도체TOP3+ | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 226380 | ACE Fn성장소비주도주 | FnGuide Growth and Consumption-Driven Stock Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0177X0 | ACE K휴머노이드로봇산업TOP2+ | FnGuide AI Semiconductor TOP2+ Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 380340 | ACE 코리아AI테크핵심산업 | FnGuide Korea AI Tech Core Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 466810 | BNK 2차전지양극재 | FnGuide Secondary Battery Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 487750 | BNK 온디바이스AI | FnGuide On-Device AI Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 486240 | DAISHIN343 AI반도체&인프라액티브 | FnGuide AI Semiconductor & Infrastructure Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0189Z0 | DAISHIN343 금융&지주고배당 | FnGuide K High Dividend Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 0174J0 | DAISHIN343 오피스리츠플러스 | FnGuide AI Semiconductor TOP2 Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 395290 | HANARO Fn K-POP&미디어 | FnGuide K-POP & Media Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 395280 | HANARO Fn K-게임 | FnGuide K-Game Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 395270 | HANARO Fn K-반도체 | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 441540 | HANARO Fn조선해운 | FnGuide Shipbuilding & Shipping Industry Index | needs_core_market_data | load_core_market_and_calendar_data |
| 381570 | HANARO Fn친환경에너지 | FnGuide Green Energy Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 498050 | HANARO 바이오코리아액티브 | FnGuide K-Bio Balanced Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 476260 | HANARO 반도체핵심공정주도주 | FnGuide Semiconductor Core Process Tech Leaders Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0111J0 | HANARO 증권고배당TOP3플러스 | FnGuide High Dividend Securities TOP3 Plus Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 0005G0 | IBK K-AI반도체코어테크 | FnGuide K-AI Semiconductor Core Technology Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 488200 | KIWOOM K-2차전지북미공급망 | FnGuide Secondary Battery Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 488210 | KIWOOM K-반도체북미공급망 | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 483020 | KIWOOM 의료AI | FnGuide AI medical technology Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 363580 | KODEX 200IT TR | FnGuide-RAFI Korea Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 305720 | KODEX 2차전지산업 | FnGuide Secondary Battery Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 461950 | KODEX 2차전지핵심소재10 | FnGuide Secondary Battery Core Materials 10 Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 395160 | KODEX AI반도체TOP2플러스 | FnGuide AI Semiconductor TOP2 Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 471990 | KODEX AI반도체핵심장비 | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 487240 | KODEX AI전력핵심설비 | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 266370 | KODEX IT | FnGuide IT PLUS Index | needs_core_market_data | load_core_market_and_calendar_data |
| 117700 | KODEX 건설 | Maekyung FnGuide Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 300950 | KODEX 게임산업 | FnGuide Game Industry Index | needs_core_market_data | load_core_market_and_calendar_data |
| 266390 | KODEX 경기소비재 | Maekyung FnGuide Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 0089D0 | KODEX 금융고배당TOP10 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 498410 | KODEX 금융고배당TOP10타겟위클리커버드콜 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 102960 | KODEX 기계장비 |  | blocked_missing_pdf | find_fnguide_methodology_pdf |
| 445290 | KODEX 로봇액티브 |  | blocked_missing_pdf | find_fnguide_methodology_pdf |
| 244580 | KODEX 바이오 | FnGuide K-Bio Balanced Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 091160 | KODEX 반도체 | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0190G0 | KODEX 반도체타겟위클리커버드콜 | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0080G0 | KODEX 방산TOP10 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 140700 | KODEX 보험 | Maekyung FnGuide Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 385510 | KODEX 신재생에너지액티브 | FnGuide K-Renewable Energy Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 117460 | KODEX 에너지화학 | Maekyung FnGuide Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 091170 | KODEX 은행 | FnGuide Bank High Dividend Plus TOP 10 Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 091180 | KODEX 자동차 | FnGuide Sector Coverage Index Series | needs_core_market_data | load_core_market_and_calendar_data |
| 0115D0 | KODEX 조선TOP10 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 102970 | KODEX 증권 | FnGuide Sector Coverage Index Series | needs_core_market_data | load_core_market_and_calendar_data |
| 117680 | KODEX 철강 |  | blocked_missing_pdf | find_fnguide_methodology_pdf |
| 445150 | KODEX 친환경조선해운액티브 | FnGuide Shipbuilding & Shipping Industry Index | needs_core_market_data | load_core_market_and_calendar_data |
| 0115E0 | KODEX 코리아소버린AI | FnGuide Korea AI Tech Core Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 266410 | KODEX 필수소비재 | Maekyung FnGuide Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 298770 | KODEX 한국대만IT프리미어 | FnGuide IT PLUS Index | needs_core_market_data | load_core_market_and_calendar_data |
| 450190 | KODEX 한중반도체(합성) | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 266420 | KODEX 헬스케어 | FnGuide Healthcare Index | needs_core_market_data | load_core_market_and_calendar_data |
| 487130 | KoAct AI인프라액티브 | FnGuide AI Semiconductor & Infrastructure Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 462900 | KoAct 바이오헬스케어액티브 | FnGuide Healthcare Index | needs_core_market_data | load_core_market_and_calendar_data |
| 482030 | KoAct 반도체&2차전지핵심소재액티브 | FnGuide Secondary Battery Core Materials 10 Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 284980 | RISE 200금융 | FnGuide-RAFI Korea Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 465330 | RISE 2차전지TOP10 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 422420 | RISE 2차전지액티브 | FnGuide Secondary Battery Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 469070 | RISE AI&로봇 | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0093A0 | RISE AI반도체TOP10 | FnGuide AI Semiconductor TOP10 Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0101N0 | RISE AI전력인프라 | FnGuide AI Semiconductor & Infrastructure Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 427120 | RISE AI플랫폼 | FnGuide AI Platform Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 326240 | RISE IT플러스 | FnGuide IT PLUS Index | needs_core_market_data | load_core_market_and_calendar_data |
| 388280 | RISE K엔터&여행레저 | FnGuide K-Entertainment & Travel Leisure Index | needs_core_market_data | load_core_market_and_calendar_data |
| 300640 | RISE 게임테마 | FnGuide K-Game Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0000Z0 | RISE 바이오TOP10액티브 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 446700 | RISE 배터리 리사이클링 | FnGuide All-Solid-State Battery & Silicon Anode Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 388420 | RISE 비메모리반도체액티브 | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 498860 | RISE 코리아금융고배당 | FnGuide KOREA High Dividend | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 253280 | RISE 헬스케어 | FnGuide Healthcare Index | needs_core_market_data | load_core_market_and_calendar_data |
| 0190C0 | RISE 현대차고정피지컬AI | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 455860 | SOL 2차전지소부장Fn | FnGuide Secondary Battery Materials & Equipment Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0167A0 | SOL AI반도체TOP2플러스 | FnGuide AI Semiconductor TOP2 Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 455850 | SOL AI반도체소부장 | FnGuide AI Semiconductor Materials & Equipment Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 490480 | SOL K방산 | FnGuide Defense TOP5 Index TR | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 484880 | SOL 금융지주플러스고배당 | FnGuide Financial Group Plus High Dividend Index(PR) | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 475300 | SOL 반도체전공정 | FnGuide Semiconductor Front-End Process Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 475310 | SOL 반도체후공정 | FnGuide Semiconductor Back-End Process Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 464610 | SOL 의료기기소부장Fn | FnGuide Medical Device Materials & Equipment Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 466930 | SOL 자동차TOP3플러스 | FnGuide Automobile TOP3 Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 464600 | SOL 자동차소부장Fn | FnGuide Automobile Materials & Equipment Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0005D0 | SOL 전고체배터리&실리콘음극재 | FnGuide All-Solid-State Battery & Silicon Anode Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 466920 | SOL 조선TOP3플러스 | FnGuide Shipbuilding TOP3 Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0141S0 | SOL 조선기자재 | FnGuide Shipbuilding Equipment Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0105D0 | SOL 한국AI소프트웨어 | FnGuide Sector Coverage Index Series | needs_core_market_data | load_core_market_and_calendar_data |
| 0008T0 | SOL 화장품TOP3플러스 | FnGuide Cosmetics TOP3 Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 315270 | TIGER 200커뮤니케이션서비스 | FnGuide-RAFI Korea Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 364980 | TIGER 2차전지TOP10 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 462010 | TIGER 2차전지소재Fn | FnGuide Secondary Battery Material Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 305540 | TIGER 2차전지테마 | FnGuide Secondary Battery Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 471760 | TIGER AI반도체핵심공정 | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 365040 | TIGER AI코리아그로스액티브 | FnGuide Korea AI Tech Core Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 377990 | TIGER Fn신재생에너지 | FnGuide K-Renewable Energy Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 300610 | TIGER K게임 | FnGuide K-Game Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 463250 | TIGER K방산&우주 | FnGuide Defense TOP5 Index TR | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 364990 | TIGER 게임TOP10 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 0168K0 | TIGER 기술이전바이오액티브 | FnGuide K-Bio Balanced Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 228810 | TIGER 미디어컨텐츠 | FnGuide K-POP & Media Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 364970 | TIGER 바이오TOP10 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 091230 | TIGER 반도체 | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 396500 | TIGER 반도체TOP10 | FnGuide AI Semiconductor TOP10 Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0177R0 | TIGER 반도체TOP10커버드콜액티브 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 157490 | TIGER 소프트웨어 | FnGuide Sector Coverage Index Series | needs_core_market_data | load_core_market_and_calendar_data |
| 091220 | TIGER 은행 | FnGuide Bank High Dividend Plus TOP 10 Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 466940 | TIGER 은행고배당플러스TOP10 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 307510 | TIGER 의료기기 | FnGuide Healthcare Equipment Index | needs_core_market_data | load_core_market_and_calendar_data |
| 365000 | TIGER 인터넷TOP10 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 494670 | TIGER 조선TOP10 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 157500 | TIGER 증권 | FnGuide Sector Coverage Index Series | needs_core_market_data | load_core_market_and_calendar_data |
| 307520 | TIGER 지주회사 | FnGuide Holdings Company Index | needs_core_market_data | load_core_market_and_calendar_data |
| 0117V0 | TIGER 코리아AI전력기기TOP3플러스 | FnGuide Automobile TOP3 Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0148J0 | TIGER 코리아휴머노이드로봇산업 | FnGuide Korea AI Tech Core Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 387280 | TIGER 퓨처모빌리티액티브 | FnGuide Fuel Cell Electric Future Mobility Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 449690 | TIGER 한중반도체(합성) | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 143860 | TIGER 헬스케어 | FnGuide Healthcare Index | needs_core_market_data | load_core_market_and_calendar_data |
| 228790 | TIGER 화장품 | FnGuide Cosmetics TOP3 Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 463050 | TIME K바이오액티브 | FnGuide K-Bio Balanced Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 404120 | TIME K신재생에너지액티브 | FnGuide K-Renewable Energy Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0184V0 | UNICORN K바이오액티브 | FnGuide K-Bio Balanced Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 470310 | UNICORN 생성형AI강소기업액티브 | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 422260 | VITA MZ소비액티브 | FnGuide MZ Consumption Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 413930 | WON AI ESG액티브 | FnGuide ESGM ESG 지수 | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 474590 | WON 반도체밸류체인액티브 | FnGuide Semiconductor Value Chain Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0154F0 | WON 초대형IB&금융지주 | FnGuide Financial Group Plus High Dividend Index(PR) | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 0001P0 | 마이티 바이오시밀러&CDMO액티브 | FnGuide Bio Similar & CDMO Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
