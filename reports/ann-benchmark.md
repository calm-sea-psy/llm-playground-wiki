# ANN 벤치마크 — ann-benchmark
재현: llm-playground-retrieval 레포에서 `QDRANT_TIMEOUT=300 uv run --group bench --extra rerank python bench/bench_ann.py --collection kowiki_kure-v1 --skip-qdrant-exact --float-ef 16 32 --qdrant-batch 10 --nlist 4096 --run-id ann-benchmark` (질문 1,000 · seed 0 은 기본값, 이 레포를 `../llm-playground-wiki` 에 둔다)

- 컬렉션: `kowiki_kure-v1` 벡터 2,393,485개 (kure-v1, 청크 384토큰) · KorQuAD dev 질문 1,000개 (정답 문서가 컬렉션에 있는 질문 중 무작위, seed 0)
- `recall_at_10`: FAISS Flat(전수 비교) 상위 10건 대비. `hit_at_5` · `mrr_at_10`: evalkit `contains_answer` 정답 판정
- FAISS: CPU, 지연은 스레드 1개 · 질의 하나씩, `qps` 는 질의 1,000개 일괄 · 스레드 24개. `index_mb` 는 직렬화한 색인 크기
- Qdrant: 로컬 Docker, HNSW m=16 · ef_construct=100 + int8 스칼라 양자화(RAM) · 원본 벡터 on_disk. 지연은 gRPC 왕복 포함 (REST 는 질의마다 약 16ms 더 든다), `qps` 는 요청 하나에 10개씩 묶어 보냄. 색인 시간 · 크기는 적재 때라 비움
- 전원 · 장비: RTX 5080 Laptop · GPU 전력 상한 95W · CPU 24스레드 (FAISS · Qdrant 측정은 CPU)
- IVF 목록 수 4096 (`--nlist 4096`): 자동값(4·√n 에 가까운 8192)은 IVF 세 종류가 각자 군집을 학습해 약 45분이 더 들어 줄였다
- Qdrant `HNSW float` 는 ef 16 · 32 만 (원본 벡터를 디스크에서 읽어 질의당 수백 ms), 전수 검색(`exact`)은 뺐다

## 읽을 점

- **규모가 커지면 같은 설정의 recall 이 크게 떨어진다.** FAISS HNSW32 ef=128: 서브셋(5만) 0.989 → 전체(239만) 0.919. Qdrant int8 ef=128: 0.969 → 0.952
- **Qdrant 는 int8 양자화(RAM)가 필수다.** 원본 벡터(on_disk)로 찾으면 질의당 248~382ms · QPS 2~4, int8 은 4~12ms
- **rescore + oversampling 2배가 전체 규모에서 효과가 크다.** ef=128 에서 recall 0.952 → 0.986, 지연은 7.6 → 8.4ms
- **hit@5 는 recall 0.95 부근부터 거의 늘지 않는다** (Flat 0.714, Qdrant int8+rescore ef=128 0.709). 지금 서빙 설정(int8, 기본 ef)으로 품질 손실은 1%p 안팎
- **IVF 는 이 규모 · 차원에서 HNSW 보다 불리하다.** nprobe=64 에서도 recall 0.879, IVF+PQ128 은 0.615 에서 멈춘다 (색인은 HNSW 의 1/30 크기)

![recall@10 – QPS](ann-benchmark.svg)

오른쪽 위로 갈수록 좋다 (같은 recall 에서 더 빠름). 각 점은 아래 표의 한 줄이다.

| engine | index | param | recall_at_10 | hit_at_5 | mrr_at_10 | p50_ms | p99_ms | qps | build_sec | index_mb |
|---|---|---|---|---|---|---|---|---|---|---|
| faiss | Flat | - | 1.000 | 0.714 | 0.580 | 212.970 | 222.470 | 168 | 2.400 | 9349.600 |
| faiss | HNSW32 | efSearch=16 | 0.693 | 0.535 | 0.431 | 0.210 | 0.470 | 15408 | 266.500 | 9970.700 |
| faiss | HNSW32 | efSearch=32 | 0.798 | 0.600 | 0.491 | 0.320 | 0.620 | 9407 | 266.500 | 9970.700 |
| faiss | HNSW32 | efSearch=64 | 0.869 | 0.640 | 0.525 | 0.550 | 1.030 | 5750 | 266.500 | 9970.700 |
| faiss | HNSW32 | efSearch=128 | 0.919 | 0.666 | 0.544 | 0.960 | 1.530 | 3393 | 266.500 | 9970.700 |
| faiss | HNSW32 | efSearch=256 | 0.951 | 0.684 | 0.557 | 1.750 | 2.560 | 1853 | 266.500 | 9970.700 |
| faiss | IVF4096,Flat | nprobe=1 | 0.282 | 0.264 | 0.219 | 0.490 | 0.700 | 11459 | 150.200 | 9402.100 |
| faiss | IVF4096,Flat | nprobe=4 | 0.511 | 0.438 | 0.364 | 0.650 | 0.910 | 5094 | 150.200 | 9402.100 |
| faiss | IVF4096,Flat | nprobe=16 | 0.730 | 0.578 | 0.474 | 1.230 | 1.510 | 1926 | 150.200 | 9402.100 |
| faiss | IVF4096,Flat | nprobe=64 | 0.879 | 0.650 | 0.536 | 4.060 | 4.740 | 506 | 150.200 | 9402.100 |
| faiss | IVF4096,SQ8 | nprobe=4 | 0.511 | 0.438 | 0.364 | 2.080 | 3.000 | 3553 | 151.200 | 2389.900 |
| faiss | IVF4096,SQ8 | nprobe=16 | 0.728 | 0.578 | 0.474 | 7.270 | 9.110 | 1165 | 151.200 | 2389.900 |
| faiss | IVF4096,SQ8 | nprobe=64 | 0.876 | 0.650 | 0.536 | 27.600 | 31.840 | 316 | 151.200 | 2389.900 |
| faiss | IVF4096,PQ128 | nprobe=4 | 0.431 | 0.423 | 0.346 | 0.610 | 0.920 | 12828 | 310.200 | 345.700 |
| faiss | IVF4096,PQ128 | nprobe=16 | 0.558 | 0.547 | 0.443 | 1.150 | 1.340 | 7771 | 310.200 | 345.700 |
| faiss | IVF4096,PQ128 | nprobe=64 | 0.615 | 0.609 | 0.491 | 3.390 | 3.840 | 3031 | 310.200 | 345.700 |
| qdrant | HNSW float | hnsw_ef=16 | 0.866 | 0.649 | 0.534 | 248.180 | 436.490 | 4 | - | - |
| qdrant | HNSW float | hnsw_ef=32 | 0.926 | 0.681 | 0.558 | 382.400 | 676.300 | 2 | - | - |
| qdrant | HNSW int8 | hnsw_ef=16 | 0.848 | 0.649 | 0.535 | 9.140 | 149.330 | 154 | - | - |
| qdrant | HNSW int8 | hnsw_ef=32 | 0.902 | 0.680 | 0.557 | 4.190 | 6.100 | 145 | - | - |
| qdrant | HNSW int8 | hnsw_ef=64 | 0.935 | 0.702 | 0.570 | 5.290 | 7.600 | 128 | - | - |
| qdrant | HNSW int8 | hnsw_ef=128 | 0.952 | 0.711 | 0.577 | 7.590 | 10.260 | 164 | - | - |
| qdrant | HNSW int8 | hnsw_ef=256 | 0.958 | 0.713 | 0.580 | 11.640 | 14.650 | 106 | - | - |
| qdrant | HNSW int8+rescore x2 | hnsw_ef=16 | 0.890 | 0.662 | 0.546 | 13.520 | 23.730 | 142 | - | - |
| qdrant | HNSW int8+rescore x2 | hnsw_ef=32 | 0.926 | 0.680 | 0.558 | 5.910 | 10.650 | 137 | - | - |
| qdrant | HNSW int8+rescore x2 | hnsw_ef=64 | 0.965 | 0.700 | 0.571 | 7.270 | 12.490 | 124 | - | - |
| qdrant | HNSW int8+rescore x2 | hnsw_ef=128 | 0.986 | 0.709 | 0.577 | 8.390 | 13.170 | 167 | - | - |
| qdrant | HNSW int8+rescore x2 | hnsw_ef=256 | 0.993 | 0.713 | 0.580 | 11.790 | 16.970 | 106 | - | - |
