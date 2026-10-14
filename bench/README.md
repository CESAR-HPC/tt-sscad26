# Benchmarks

Ainda não implementado. Os exemplos GEMM (05–08) imprimirão uma linha CSV

```
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
```

que será coletada em `bench/results/`. As medições de desempenho são feitas em
hardware real; no simulador, os tempos não são representativos.
