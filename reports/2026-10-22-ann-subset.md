# ANN 벤치마크 — 2026-10-22-ann-subset
재현: llm-playground-retrieval 레포에서 `uv run --group bench --extra rerank python bench/bench_ann.py --collection kowiki-subset-10000_kure-v1 --run-id 2026-10-22-ann-subset` (질문 1,000 · seed 0 · nlist 자동 1024 · 묶음 100 은 기본값, 이 레포를 `../llm-playground-wiki` 에 둔다)

- 컬렉션: `kowiki-subset-10000_kure-v1` 벡터 51,005개 (kure-v1, 청크 384토큰) · KorQuAD dev 질문 1,000개 (정답 문서가 컬렉션에 있는 질문 중 무작위, seed 0)
- `recall_at_10`: FAISS Flat(전수 비교) 상위 10건 대비. `hit_at_5` · `mrr_at_10`: evalkit `contains_answer` 정답 판정
- FAISS: CPU, 지연은 스레드 1개 · 질의 하나씩, `qps` 는 질의 1,000개 일괄 · 스레드 24개. `index_mb` 는 직렬화한 색인 크기
- Qdrant: 로컬 Docker, HNSW m=16 · ef_construct=100 + int8 스칼라 양자화(RAM) · 원본 벡터 on_disk. 지연은 gRPC 왕복 포함 (REST 는 질의마다 약 16ms 더 든다), `qps` 는 요청 하나에 100개씩 묶어 보냄. 색인 시간 · 크기는 적재 때라 비움
- **서브셋 1차 확인용** — 전체 규모(약 230만 벡터)는 Phase 1.5

| engine | index | param | recall_at_10 | hit_at_5 | mrr_at_10 | p50_ms | p99_ms | qps | build_sec | index_mb |
|---|---|---|---|---|---|---|---|---|---|---|
| faiss | Flat | - | 1.000 | 0.865 | 0.782 | 4.490 | 4.860 | 5845 | 0.100 | 199.200 |
| faiss | HNSW32 | efSearch=16 | 0.910 | 0.816 | 0.742 | 0.120 | 0.250 | 25985 | 3.400 | 212.500 |
| faiss | HNSW32 | efSearch=32 | 0.950 | 0.840 | 0.764 | 0.200 | 0.350 | 16200 | 3.400 | 212.500 |
| faiss | HNSW32 | efSearch=64 | 0.976 | 0.854 | 0.774 | 0.340 | 0.530 | 10160 | 3.400 | 212.500 |
| faiss | HNSW32 | efSearch=128 | 0.989 | 0.858 | 0.777 | 0.600 | 0.880 | 5766 | 3.400 | 212.500 |
| faiss | HNSW32 | efSearch=256 | 0.997 | 0.864 | 0.781 | 1.080 | 1.550 | 3116 | 3.400 | 212.500 |
| faiss | IVF1024,Flat | nprobe=1 | 0.498 | 0.507 | 0.465 | 0.080 | 0.090 | 72664 | 2.000 | 204.000 |
| faiss | IVF1024,Flat | nprobe=4 | 0.722 | 0.683 | 0.623 | 0.100 | 0.120 | 45358 | 2.000 | 204.000 |
| faiss | IVF1024,Flat | nprobe=16 | 0.875 | 0.777 | 0.711 | 0.180 | 0.220 | 17363 | 2.000 | 204.000 |
| faiss | IVF1024,Flat | nprobe=64 | 0.958 | 0.842 | 0.765 | 0.470 | 0.570 | 5359 | 2.000 | 204.000 |
| faiss | IVF1024,SQ8 | nprobe=4 | 0.721 | 0.683 | 0.623 | 0.250 | 0.360 | 30075 | 2.300 | 54.600 |
| faiss | IVF1024,SQ8 | nprobe=16 | 0.873 | 0.777 | 0.711 | 0.750 | 0.970 | 9427 | 2.300 | 54.600 |
| faiss | IVF1024,SQ8 | nprobe=64 | 0.955 | 0.842 | 0.765 | 2.580 | 2.980 | 2938 | 2.300 | 54.600 |
| faiss | IVF1024,PQ128 | nprobe=4 | 0.639 | 0.674 | 0.594 | 0.170 | 0.530 | 44703 | 154.400 | 12.000 |
| faiss | IVF1024,PQ128 | nprobe=16 | 0.724 | 0.765 | 0.674 | 0.230 | 0.310 | 36281 | 154.400 | 12.000 |
| faiss | IVF1024,PQ128 | nprobe=64 | 0.751 | 0.815 | 0.719 | 0.440 | 0.500 | 17980 | 154.400 | 12.000 |
| qdrant | exact float | - | 1.000 | 0.865 | 0.782 | 7.620 | 9.010 | 374 | - | - |
| qdrant | HNSW float | hnsw_ef=16 | 0.893 | 0.825 | 0.750 | 1.690 | 2.650 | 1369 | - | - |
| qdrant | HNSW float | hnsw_ef=32 | 0.945 | 0.842 | 0.764 | 1.780 | 2.480 | 1309 | - | - |
| qdrant | HNSW float | hnsw_ef=64 | 0.975 | 0.852 | 0.772 | 1.950 | 2.780 | 1159 | - | - |
| qdrant | HNSW float | hnsw_ef=128 | 0.991 | 0.862 | 0.780 | 2.250 | 2.940 | 1018 | - | - |
| qdrant | HNSW float | hnsw_ef=256 | 0.996 | 0.864 | 0.782 | 2.800 | 3.670 | 890 | - | - |
| qdrant | HNSW int8 | hnsw_ef=16 | 0.880 | 0.827 | 0.749 | 1.640 | 2.440 | 1446 | - | - |
| qdrant | HNSW int8 | hnsw_ef=32 | 0.929 | 0.844 | 0.763 | 1.690 | 2.260 | 1399 | - | - |
| qdrant | HNSW int8 | hnsw_ef=64 | 0.956 | 0.854 | 0.772 | 1.810 | 2.530 | 1260 | - | - |
| qdrant | HNSW int8 | hnsw_ef=128 | 0.969 | 0.865 | 0.781 | 2.020 | 2.890 | 1208 | - | - |
| qdrant | HNSW int8 | hnsw_ef=256 | 0.974 | 0.867 | 0.783 | 2.460 | 4.080 | 1050 | - | - |
| qdrant | HNSW int8+rescore x2 | hnsw_ef=16 | 0.909 | 0.834 | 0.758 | 1.670 | 2.430 | 1394 | - | - |
| qdrant | HNSW int8+rescore x2 | hnsw_ef=32 | 0.944 | 0.841 | 0.762 | 1.700 | 2.340 | 1367 | - | - |
| qdrant | HNSW int8+rescore x2 | hnsw_ef=64 | 0.975 | 0.851 | 0.771 | 1.820 | 2.570 | 1324 | - | - |
| qdrant | HNSW int8+rescore x2 | hnsw_ef=128 | 0.991 | 0.862 | 0.780 | 2.050 | 2.700 | 1157 | - | - |
| qdrant | HNSW int8+rescore x2 | hnsw_ef=256 | 0.997 | 0.864 | 0.782 | 2.440 | 3.230 | 723 | - | - |
