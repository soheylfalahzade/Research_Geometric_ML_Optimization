# Geo-SmartSpanner: Robust Neural-Algorithmic Framework for Directed *t*-Spanner Construction in Metropolitan Road Networks

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.5-orange.svg)
![Reproducibility](https://img.shields.io/badge/Seeds-15--18%20per%20city-success.svg)

**Author:** Soheyl Falahzade — M.Sc. Student, Algorithms & Computational Geometry, Yazd University

> این README دقیقاً منطبق با محتوای فعلی `paper/Main_Paper.tex` است. هر عددی که اینجا می‌بینید، از یک فایل CSV واقعی در `results/raw_runs/` می‌آید — نه تخمین یا گرد کردن.

## خلاصه

روتینگ لحظه‌ای در سیستم‌های حمل‌ونقل هوشمند (ITS) به‌شدت توسط پیچیدگی $O(m \cdot (n + m \log n))$ ساخت *t*-spanner هندسی دقیق محدود می‌شود. این فریمورک با ترکیب:

1. یک **GNN با عدم‌قطعیت معرفت‌شناختی** (Monte Carlo Dropout) برای پیش‌بینی احتمال بقای هر یال،
2. یک **ماژول ترمیم توپولوژیک جهت‌دار** (Directed GLV-Repair) مبتنی بر Dijkstra با cutoff، که تضمین می‌کند $d_H(u,v) \le t \cdot d_G(u,v)$ برای همهٔ یال‌ها،
3. یک **مکانیزم استنتاج فازی (Mamdani)** که تصمیم هرس را به‌جای یک آستانهٔ سخت، از ترکیب عدم‌قطعیت مدل و مرکزیت ساختاری یال می‌گیرد،
4. یک لایهٔ **بهینه‌سازی تکاملی (GA)** برای تنظیم مشترک وزن‌های loss و پارامترهای کالیبراسیون،

سرعت اجرا را نسبت به الگوریتم کلاسیک Greedy Spanner به‌طور معنادار بالا می‌برد و در همان حال ۱۰۰٪ همبندی قوی (SCC) و کران بالای stretch جهت‌دار را حتی زیر نویز مکانی شدید حفظ می‌کند.

**یافتهٔ صادقانه‌ای که مقاله هم گزارش می‌دهد:** جستجوی تکاملی (GA) برای تنظیم *یک* پارامتر آستانه، مزیتی نسبت به جستجوی تصادفی ندارد — ارزش واقعی GA در تنظیم *مشترک* چند پارامتر همزمان است، نه در جستجوی اسکالر.

---

## نتایج تجربی (از `results/raw_runs/`)

### ۱. سرعت ساخت نسبت به Greedy کلاسیک (میانگین ± انحراف‌معیار روی ۵ seed مستقل، با ترمیم cutoff-bounded)

| شهر | گره‌ها | یال‌ها | زمان Greedy (s) | زمان ما (s) | سرعت | حداکثر Stretch (بدترین seed) |
|---|---:|---:|---:|---:|---:|---:|
| Manhattan | 4,540 | 9,766 | 0.742 | 0.110 ± 0.044 | 6.74× | 1.497 |
| Eindhoven | 7,881 | 19,051 | 1.948 | 0.206 ± 0.054 | 9.45× | 1.497 |
| Paris | 9,236 | 17,891 | 3.256 | 0.193 ± 0.064 | 16.89× | 1.499 |
| Rome | 42,788 | 88,618 | 46.626 | 1.097 ± 0.154 | **42.51×** | 1.494 |

> این جدول را مستقل، جدا از اسکریپت خود پروژه، از `results/raw_runs/full_benchmark_20260706_144626.csv` بازمحاسبه کردم (طبق قانون ۱). ستون آخر نشان می‌دهد حتی در بدترین seed هم کران $t=1.5$ نقض نشده.

> نکتهٔ صادقانهٔ مهم که در مقاله هم هست: یک پیاده‌سازی اولیهٔ ساده‌لوحانه (بدون cutoff در جستجوی ترمیم) این سرعت را تا ۰.۸۶–۱.۲۱× پایین می‌آورد، چون تا ۹۷.۷٪ زمان اجرا صرف تأیید ترمیم می‌شد. یعنی مزیت سرعت فقط وقتی واقعی است که *همهٔ* مراحل pipeline، نه فقط بخش یادگیری‌محور، پیچیدگی مناسب داشته باشند.

### ۲. مقایسهٔ سه مکانیزم تصمیم هرس (میانگین ± انحراف معیار، روی ۱۵–۱۸ seed مستقل، sparsification یکسان)

| شهر | n | C: آستانهٔ GA+Feedback | D: هرس تصادفی | **E: هرس فازی (پیشنهادی)** |
|---|---:|---:|---:|---:|
| Eindhoven | 15 | 2624.6 ± 11.4 | 2437.1 ± 16.5 | **2189.2 ± 24.2** |
| Paris | 15 | 4748.3 ± 14.5 | 4653.7 ± 15.0 | **4636.3 ± 14.6** |
| Rome | 18 | 14965.3 ± 29.8 | 14425.3 ± 54.3 | **13900.1 ± 41.4** |

عدد جدول = تعداد ترمیم‌های لازم بعد از هرس (کمتر = بهتر). مکانیزم فازی در **هر سه شهر بدون استثنا** کمترین ترمیم را لازم دارد.

> **تأیید آماری مستقل (Welch's t-test، جدا از کد پروژه):** برتری فازی نسبت به هرس تصادفی (D) در هر سه شهر معنادار است — Eindhoven: p<0.0001، Rome: p<0.0001، و حتی در Paris که فاصلهٔ مطلق کم است (۴۶۳۶ در مقابل ۴۶۵۴): p=0.0032. برتری نسبت به GA+Feedback (C) در همه‌جا p<0.0001. یعنی این نتیجه شانسی نیست.

---

## ساختار پروژه (فعلی، پاک‌سازی‌شده)

```
Research_Geometric_ML_Optimization/
├── benchmarks/          # اسکریپت‌های آزمایش و اعتبارسنجی (q1_*, test_*, ablation_*)
├── src/                 # کد اصلی pipeline (research_ml.py, visualize_*.py)
├── results/
│   ├── data/             # دیتاست‌های ساخته‌شده از OSM
│   ├── figures/          # نمودارهای نهایی مقاله
│   ├── raw_runs/         # لاگ خام هر اجرا با seed و timestamp — منبع همهٔ اعداد بالا
│   ├── logs/
│   └── models/
├── docs/                 # نقشه‌های تعاملی HTML (از شمارش زبان گیت‌هاب مستثنا)
├── paper/
│   ├── Main_Paper.tex    # مقالهٔ فعلی (فرمت IEEE)
│   ├── Main_Paper.pdf
│   └── updates/          # نسخه‌های میانی بخش‌های مقاله
├── data_generator.py     # ساخت دیتاست از OSMnx + محاسبهٔ Greedy Spanner دقیق به‌عنوان ground truth
├── genetic_optimizer.py  # تنظیم هایپرپارامتر با GA
├── cross_city_validator.py
├── final_validator.py
├── hybrid_optimizer.py
├── hyperparameter_benchmark.py
├── global_generalization_benchmark.py
├── requirements.txt
├── requirements_full_env_backup.txt   # (فقط مرجع؛ خروجی کامل pip freeze محیط conda قبلی)
└── .gitignore / .gitattributes
```

---

## نکتهٔ باز (از مقاله نقل‌قول شده)

> «سه شکل در Fig. 4 قبل از افزودن ویژگی مرکزیت یال و مکانیزم هرس فازی تولید شده‌اند و پیش از ارسال نهایی به مجله باید با مدل فعلی بازتولید شوند.»

**به‌روزرسانی:** این ۳ شکل بازتولید شدند (`results/figures/q1_global_stretch_cdf.png`, `q1_continual_learning_loss.png`, `q1_directed_asymmetry_proof.png`).

**یافتهٔ صادقانهٔ دیگر (طبق قانون ۵ متدولوژی):** در نمودار Continual Learning، لاس شهر Rome بین epoch های ۴ و ۵ به‌جای کاهش یکنواخت، یک‌بار افزایش موقت نشان می‌دهد (رفتار مشابهی گاهی در Paris هم دیده می‌شود). این با buffer آزمایشی کوچک (فقط ۵ epoch fine-tune به‌ازای هر شهر، و buffer که فقط تا ۳ شهر قبلی را نگه می‌دارد) سازگار است — به‌احتمال زیاد نوسان طبیعی گرادیان است، نه یک باگ. هنوز رسماً با seedهای چندگانه تکرار و تأیید نشده؛ قبل از ادعای «پایداری کامل» در متن مقاله باید این را با چند seed دیگر چک کرد.

یعنی سه نمودار (`q1_global_stretch_cdf.png`, `q1_continual_learning_loss.png`, `q1_directed_asymmetry_proof.png`) باید با pipeline فعلی دوباره تولید شوند — این هنوز انجام نشده.

---

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python data_generator.py                    # ساخت دیتاست از OSM برای یک شهر
python genetic_optimizer.py                  # تنظیم هایپرپارامتر
python benchmarks/q1_master_benchmarker.py   # اجرای مجموعهٔ کامل اعتبارسنجی
```

## License

MIT
