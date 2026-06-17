# FnGuide first-pass coverage

- ETFs covered: 142
- Methodology PDFs available: 137
- Methodology PDFs missing: 5

## Readiness

- needs_external_model_data: 66
- needs_core_market_data: 43
- needs_dividend_or_custom_data: 28
- blocked_missing_pdf: 5

## Families

- keyword_theme: 54
- dividend: 27
- equal_weight: 19
- sector_theme: 18
- theme: 10
- top_n_theme: 9
- unknown: 5

## Missing Data

- constituent_universe: 142
- corporate_actions: 142
- krx_trading_calendar: 142
- listed_shares: 142
- stock_prices: 142
- market_cap: 128
- free_float_ratio: 127
- ranking_inputs: 108
- trading_value_liquidity: 107
- futures_options_expiry_calendar: 95
- fics_industry_classification: 65
- keyword_source_documents: 64
- dividend_data: 27
- custom_index_formula_inputs: 8
- methodology_pdf_missing: 5
- score_inputs: 5
- month_end_business_day_calendar: 3
- fundamental_factors: 1

## ETF Actions

| code | name | index | readiness | next action |
| --- | --- | --- | --- | --- |
| 396500 | TIGER 반도체TOP10 | FnGuide AI Semiconductor TOP10 Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0167A0 | SOL AI반도체TOP2플러스 | FnGuide AI Semiconductor TOP2 Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 395270 | HANARO Fn K-반도체 | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 395160 | KODEX AI반도체TOP2플러스 | FnGuide AI Semiconductor TOP2 Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 161510 | PLUS 고배당주 | FnGuide High Dividend Stocks Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 292150 | TIGER 코리아TOP10 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 305720 | KODEX 2차전지산업 | FnGuide Secondary Battery Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 315930 | KODEX Top5PlusTR | FnGuide Defense TOP5 Index TR | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 466920 | SOL 조선TOP3플러스 | FnGuide Shipbuilding TOP3 Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 488080 | TIGER 반도체TOP10레버리지 | FnGuide Semiconductor TOP10 Leverage Index TR | needs_core_market_data | load_core_market_and_calendar_data |
| 329200 | TIGER 리츠부동산인프라 | FnGuide REITs Real Estate Infrastructure Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 449450 | PLUS K방산 | FnGuide Defense TOP5 Index TR | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 477080 | RISE CD금리액티브(합성) | FnGuide CD Interest Rates Index | needs_core_market_data | load_core_market_and_calendar_data |
| 469150 | ACE AI반도체TOP3+ | FnGuide AI Semiconductor TOP3 Plus Price Return Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 455850 | SOL AI반도체소부장 | FnGuide AI Semiconductor Materials & Equipment Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 367760 | RISE 네트워크인프라 | FnGuide Network Infrastructure Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0177R0 | TIGER 반도체TOP10커버드콜액티브 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 466940 | TIGER 은행고배당플러스TOP10 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 462330 | KODEX 2차전지산업레버리지 | FnGuide Secondary Battery Industry Leverage Index TR | needs_core_market_data | load_core_market_and_calendar_data |
| 462010 | TIGER 2차전지소재Fn | FnGuide Secondary Battery Material Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 251600 | PLUS 고배당주채권혼합 | FnGuide High Dividend Balanced Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 0105E0 | SOL 코리아고배당 | FnGuide KOREA High Dividend | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 447770 | TIGER 테슬라채권혼합Fn | FnGuide Tesla Balanced Index | needs_core_market_data | load_core_market_and_calendar_data |
| 466930 | SOL 자동차TOP3플러스 | FnGuide Automobile TOP3 Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0019K0 | TIME 미국나스닥100채권혼합50액티브 | FnGuide Nasdaq100 Balanced 50 Index | needs_core_market_data | load_core_market_and_calendar_data |
| 385510 | KODEX 신재생에너지액티브 | FnGuide K-Renewable Energy Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 435420 | TIGER 미국나스닥100채권혼합50 | FnGuide Nasdaq100 Balanced 50 Index | needs_core_market_data | load_core_market_and_calendar_data |
| 474590 | WON 반도체밸류체인액티브 | FnGuide Semiconductor Value Chain Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 307520 | TIGER 지주회사 | FnGuide Holdings Company Index | needs_core_market_data | load_core_market_and_calendar_data |
| 279530 | KODEX 고배당주 | FnGuide High Dividend Stocks Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 0180V0 | ACE 미국우주테크액티브 |  | blocked_missing_pdf | find_fnguide_methodology_pdf |
| 157500 | TIGER 증권 | FnGuide Sector Coverage Index Series | needs_core_market_data | load_core_market_and_calendar_data |
| 484880 | SOL 금융지주플러스고배당 | FnGuide Financial Group Plus High Dividend Index(PR) | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 472170 | TIGER 미국테크TOP10채권혼합 | FnGuide 미국테크TOP10 채권혼합지수 | needs_core_market_data | load_core_market_and_calendar_data |
| 0005G0 | IBK K-AI반도체코어테크 | FnGuide K-AI Semiconductor Core Technology Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 266160 | RISE 고배당 | FnGuide KQ High Dividend Focus Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 329650 | KODEX TRF3070 | FnGuide TRF Index Series | needs_core_market_data | load_core_market_and_calendar_data |
| 244580 | KODEX 바이오 | FnGuide K-Bio Balanced Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0093A0 | RISE AI반도체TOP10 | FnGuide AI Semiconductor TOP10 Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 461950 | KODEX 2차전지핵심소재10 | FnGuide Secondary Battery Core Materials 10 Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 494220 | UNICORN SK하이닉스밸류체인액티브 | FnGuide SK hynix Value Chain Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0080Y0 | SOL 조선TOP3플러스레버리지 | FnGuide Shipbuilding TOP3 Plus Leverage(2x) Index | needs_core_market_data | load_core_market_and_calendar_data |
| 475300 | SOL 반도체전공정 | FnGuide Semiconductor Front-End Process Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 325010 | KODEX 성장주 |  | blocked_missing_pdf | find_fnguide_methodology_pdf |
| 0005D0 | SOL 전고체배터리&실리콘음극재 | FnGuide All-Solid-State Battery & Silicon Anode Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 377990 | TIGER Fn신재생에너지 | FnGuide K-Renewable Energy Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0183V0 | KIWOOM 삼성전자&SK하이닉스채권혼합50 | FnGuide Samsung Electronics & SK Hynix Bond Balanced Index | needs_core_market_data | load_core_market_and_calendar_data |
| 458210 | KIWOOM CD금리액티브(합성) | FnGuide CD Interest Rates Index | needs_core_market_data | load_core_market_and_calendar_data |
| 367770 | RISE 수소경제테마 | FnGuide Golf Theme Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0092B0 | SOL 한국원자력SMR | FnGuide KOREA Nuclear Power SMR Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 325020 | KODEX 배당가치 | FnGuide SLV Dividend Value Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 457990 | PLUS 태양광&ESS | FnGuide Solar & ESS Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 326240 | RISE IT플러스 | FnGuide IT PLUS Index | needs_core_market_data | load_core_market_and_calendar_data |
| 494330 | ACE 라이프자산주주가치액티브 | FnGuide K-Shareholder Value Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 0104H0 | KoAct 미국나스닥채권혼합50액티브 | FnGuide Nasdaq100 Balanced 50 Index | needs_core_market_data | load_core_market_and_calendar_data |
| 455860 | SOL 2차전지소부장Fn | FnGuide Secondary Battery Materials & Equipment Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0000J0 | PLUS 한화그룹주 | FnGuide Hanwha Group Index | needs_core_market_data | load_core_market_and_calendar_data |
| 469170 | ACE 포스코그룹포커스 | FnGuide POSCO Group Focus Index | needs_core_market_data | load_core_market_and_calendar_data |
| 486240 | DAISHIN343 AI반도체&인프라액티브 | FnGuide AI Semiconductor & Infrastructure Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 157490 | TIGER 소프트웨어 | FnGuide Sector Coverage Index Series | needs_core_market_data | load_core_market_and_calendar_data |
| 441540 | HANARO Fn조선해운 | FnGuide Shipbuilding & Shipping Industry Index | needs_core_market_data | load_core_market_and_calendar_data |
| 329660 | KODEX TRF5050 | FnGuide TRF Index Series | needs_core_market_data | load_core_market_and_calendar_data |
| 329670 | KODEX TRF7030 | FnGuide TRF Index Series | needs_core_market_data | load_core_market_and_calendar_data |
| 475310 | SOL 반도체후공정 | FnGuide Semiconductor Back-End Process Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0098N0 | PLUS 자사주매입고배당주 | FnGuide High Dividend Stocks Plus Buyback Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 0111J0 | HANARO 증권고배당TOP3플러스 | FnGuide High Dividend Securities TOP3 Plus Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 0141S0 | SOL 조선기자재 | FnGuide Shipbuilding Equipment Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 445150 | KODEX 친환경조선해운액티브 | FnGuide Shipbuilding & Shipping Industry Index | needs_core_market_data | load_core_market_and_calendar_data |
| 381560 | HANARO Fn전기&수소차 | FnGuide Electric & Hydrogen Vehicle Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 401470 | KODEX 메타버스액티브 | FnGuide K-Metaverse MZ Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0008T0 | SOL 화장품TOP3플러스 | FnGuide Cosmetics TOP3 Plus Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 150460 | TIGER 중국소비테마 | FnGuide Golf Theme Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 473590 | ACE 미국주식베스트셀러 | FnGuide US Equity BestSeller Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 476260 | HANARO 반도체핵심공정주도주 | FnGuide Semiconductor Core Process Tech Leaders Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 147970 | TIGER 모멘텀 | FnGuide Momentum Index | needs_core_market_data | load_core_market_and_calendar_data |
| 442090 | 에셋플러스 코리아대장장이액티브 | FnGuide Blacksmith Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 400970 | TIGER Fn메타버스 | FnGuide K-Metaverse MZ Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 322410 | HANARO K고배당 | FnGuide KQ High Dividend Focus Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 447430 | ACE 주주환원가치주액티브 | FnGuide Korea All-round Value Index | needs_core_market_data | load_core_market_and_calendar_data |
| 0178H0 | IBK 미국AI TOP10국채혼합50 | FnGuide 미국AITOP10 지수 | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 380340 | ACE 코리아AI테크핵심산업 | FnGuide Korea AI Tech Core Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 395170 | KODEX Top10동일가중 | FnGuide Size/Style Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 322400 | HANARO e커머스 | FnGuide E-Commerce Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 464600 | SOL 자동차소부장Fn | FnGuide Automobile Materials & Equipment Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 385520 | KODEX 자율주행액티브 |  | blocked_missing_pdf | find_fnguide_methodology_pdf |
| 395290 | HANARO Fn K-POP&미디어 | FnGuide K-POP & Media Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 241390 | RISE V&S셀렉트밸류채권혼합 | FnGuide Select Value Balanced Index | needs_core_market_data | load_core_market_and_calendar_data |
| 0152E0 | SOL 배당성향탑픽액티브 | FnGuide Dividend Payout Top Picks | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 145850 | TREX 펀더멘탈 200 | FnGuide-RAFI Korea Index Series | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 0150K0 | KoAct 수소전력ESS인프라액티브 | FnGuide Hydrogen Power ESS Infra Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 388280 | RISE K엔터&여행레저 | FnGuide K-Entertainment & Travel Leisure Index | needs_core_market_data | load_core_market_and_calendar_data |
| 0104G0 | PLUS K방산레버리지 | FnGuide Secondary Battery Industry Leverage Index TR | needs_core_market_data | load_core_market_and_calendar_data |
| 367740 | HANARO Fn5G산업 | FnGuide 5G Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 368190 | HANARO Fn K-뉴딜디지털플러스 | FnGuide K-NewDeal Digital Plus Index | needs_core_market_data | load_core_market_and_calendar_data |
| 300950 | KODEX 게임산업 | FnGuide Game Industry Index | needs_core_market_data | load_core_market_and_calendar_data |
| 337120 | KODEX 멀티팩터 | FnGuide Multi-Factor Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 385600 | ACE 2차전지&친환경차액티브 | FnGuide Secondary Battery Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 479850 | HANARO K-뷰티 | FnGuide K-Beauty Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 445690 | BNK 주주가치액티브 | FnGuide K-Shareholder Value Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 387280 | TIGER 퓨처모빌리티액티브 | FnGuide Fuel Cell Electric Future Mobility Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 253290 | RISE 헬스케어채권혼합 | FnGuide Healthcare Balanced Index | needs_core_market_data | load_core_market_and_calendar_data |
| 491510 | 파워 K-주주가치액티브 | FnGuide K-Shareholder Value Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 244620 | KODEX 모멘텀Plus | FnGuide Momentum Index | needs_core_market_data | load_core_market_and_calendar_data |
| 410870 | TIME K컬처액티브 | FnGuide K-Culture Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 373490 | KODEX 코리아혁신성장액티브 | FnGuide KOREA High Dividend | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 280920 | PLUS 주도업종 | FnGuide Leading Industry Index | needs_core_market_data | load_core_market_and_calendar_data |
| 447660 | PLUS 애플채권혼합 | FnGuide APPLE Balanced Index | needs_core_market_data | load_core_market_and_calendar_data |
| 0153P0 | ACE 리츠부동산인프라액티브 | FnGuide REITs Real Estate Infrastructure Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 487750 | BNK 온디바이스AI | FnGuide On-Device AI Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 234310 | RISE V&S셀렉트밸류 | FnGuide Select Value Index | needs_core_market_data | load_core_market_and_calendar_data |
| 395760 | PLUS ESG성장주액티브 | FnGuide ESGM ESG 지수 | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 464610 | SOL 의료기기소부장Fn | FnGuide Medical Device Materials & Equipment Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 438900 | HANARO Fn K-푸드 | FnGuide K-Food Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 407820 | 에셋플러스 코리아플랫폼액티브 | FnGuide AI Platform Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0138D0 | RISE 동학개미 | FnGuide Investor’s choice KR Index | needs_core_market_data | load_core_market_and_calendar_data |
| 251590 | PLUS 고배당저변동50 | FnGuide High Dividend Low Volatility 50 Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 381570 | HANARO Fn친환경에너지 | FnGuide Green Energy Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 0001P0 | 마이티 바이오시밀러&CDMO액티브 | FnGuide Bio Similar & CDMO Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 174350 | TIGER 로우볼 |  | blocked_missing_pdf | find_fnguide_methodology_pdf |
| 0120J0 | BNK 카카오그룹포커스 | FnGuide KAKAO Group Focus Index | needs_core_market_data | load_core_market_and_calendar_data |
| 395750 | PLUS ESG가치주액티브 | FnGuide ESGM ESG 지수 | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 281990 | RISE 중소형고배당 | FnGuide K High Dividend Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 395150 | KODEX 웹툰&드라마 | FnGuide Webtoon and Drama Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 402460 | HANARO Fn K-메타버스MZ | FnGuide K-Metaverse MZ Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 270800 | RISE KQ고배당 | FnGuide KQ High Dividend Focus Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 395280 | HANARO Fn K-게임 | FnGuide K-Game Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 253280 | RISE 헬스케어 | FnGuide Healthcare Index | needs_core_market_data | load_core_market_and_calendar_data |
| 0184V0 | UNICORN K바이오액티브 | FnGuide K-Bio Balanced Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 227570 | TIGER 우량가치 |  | blocked_missing_pdf | find_fnguide_methodology_pdf |
| 326230 | RISE 내수주플러스 | FnGuide Domestic Consumption Plus Index | needs_core_market_data | load_core_market_and_calendar_data |
| 368680 | KODEX K-뉴딜디지털플러스 | FnGuide K-NewDeal Digital Plus Index | needs_core_market_data | load_core_market_and_calendar_data |
| 429740 | PLUS K리츠 | FnGuide REITs Index | needs_core_market_data | load_core_market_and_calendar_data |
| 244670 | KODEX 밸류Plus | FnGuide Plus Index Series | needs_core_market_data | load_core_market_and_calendar_data |
| 407300 | HANARO Fn골프테마 | FnGuide Golf Theme Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 307510 | TIGER 의료기기 | FnGuide Healthcare Equipment Index | needs_core_market_data | load_core_market_and_calendar_data |
| 266550 | PLUS 중형주저변동50 | FnGuide High Dividend Low Volatility 50 Index | needs_dividend_or_custom_data | collect_dividend_and_custom_formula_inputs |
| 483020 | KIWOOM 의료AI | FnGuide AI medical technology Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 244660 | KODEX 퀄리티Plus | FnGuide Plus Index Series | needs_core_market_data | load_core_market_and_calendar_data |
| 314700 | HANARO 농업융복합산업 | FnGuide Agriculture Business Industry Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 226380 | ACE Fn성장소비주도주 | FnGuide Growth and Consumption-Driven Stock Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 427120 | RISE AI플랫폼 | FnGuide AI Platform Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
| 422260 | VITA MZ소비액티브 | FnGuide MZ Consumption Index | needs_external_model_data | collect_theme_keyword_and_score_inputs |
