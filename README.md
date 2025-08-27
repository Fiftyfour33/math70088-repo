To use method A:

```
python a.py --csv 600601.SH.csv --date-col date --price-col close --horizon 1 --loss-mode drawdown --theta-pct 0.05 --train-frac 0.8 --alert-rate 0.15 --outdir A
```

To use method B:

```
python b.py --data-folder . --horizon 3 --loss-mode drawdown --theta-pct 0.05 --model xgb --gpu --alert-rate 0.05 --n-splits 5 --test-symbols-per-fold 2 --outdir B --cache-parquet panel.parquet --rebuild-cache B
```

To use method C:

```
python c.py --data-folder . --anchors-csv anchors.csv --anchor-cols epu,tpu,usdcny,reer,interbank_90d --anchor-lag 0 --anchor-lags 0 --anchor-bins 3 --horizon 3 --loss-mode drawdown --theta-pct 0.05 --mode xgb_anchors --gpu --n-splits 5 --test-symbols-per-fold 2 --alert-rate 0.05 --cache-parquet panel.parquet --outdir C
```
