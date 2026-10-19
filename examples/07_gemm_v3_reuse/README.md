# 07 — GEMM v3 (reuso em L1)

`C = AB`, com os tipos e tamanhos dos exemplos 05 e 06 (`A` e `B` em
bfloat16, `C` em float, acumulação em FP32). Cada núcleo calcula um **bloco**
de `C`, e não tiles isolados: blocos de `A` e de `B` ficam na L1 e cada tile é
usado várias vezes antes de ser descartado.

## Decomposição em blocos

| Parâmetro | Valor | Significado |
|---|---|---|
| `per_core_M x per_core_N` | 2 x 2 tiles | bloco de `C` de cada núcleo |
| `in0_block_w` | 2 tiles | passo em `K` (`num_k_blocks = Kt / in0_block_w = 6`) |
| `out_subblock_h x out_subblock_w` | 1 x 2 tiles | sub-bloco acumulado no Dst |

O bloco `(by, bx)` de `C` fica no núcleo lógico `(x = bx, y = by)`: são
`Nt/per_core_N x Mt/per_core_M = 8 x 9 = 72` núcleos, a grade inteira do
simulador. A cada passo `kb`:

1. o **reader** lê um bloco de `A` (`per_core_M x in0_block_w` tiles) e um de
   `B` (`in0_block_w x per_core_N`), com uma barreira para os dois;
2. o **compute** percorre os sub-blocos do bloco de `C`. Para cada um, recarrega
   no Dst a soma parcial do passo anterior (CB `c_24`), soma
   `A_blk · B_blk` com `matmul_tiles` e empacota o resultado de volta em `c_24`
   ou, no último passo, em `c_16`;
3. o **writer** escreve cada sub-bloco de `c_16` na sua posição em `C`.

O sub-bloco tem no máximo 4 tiles porque, com `fp32_dest_acc_en`, cada metade
do Dst (uma com o MATH, outra com o PACK) guarda 4 tiles FP32.

## Reuso

| | v1/v2 (05, 06) | v3 (07) |
|---|---|---|
| Leituras de cada tile de `A` da DRAM | `Nt = 16` | `Nt / per_core_N = 8` |
| Leituras de cada tile de `B` da DRAM | `Mt = 18` | `Mt / per_core_M = 9` |
| Usos de um tile de `A` em L1 | 1 | `per_core_N = 2` |
| Usos de um tile de `B` em L1 | 1 | `per_core_M = 2` |

Blocos maiores reaproveitam mais, mas usam menos núcleos e mais L1. Ainda
assim, os núcleos de uma mesma linha da grade leem da DRAM o mesmo bloco de
`A`, e os de uma mesma coluna, o mesmo bloco de `B`: é o que o exemplo 08
elimina com multicast.

## CBs

| CB | Formato | Tamanho | Conteúdo |
|---|---|---|---|
| `c_0` | bfloat16 | `2 · per_core_M · in0_block_w` tiles | dois blocos de A (double buffering) |
| `c_1` | bfloat16 | `2 · in0_block_w · per_core_N` tiles | dois blocos de B |
| `c_16` | float | `per_core_M · per_core_N` tiles | bloco de C |
| `c_24` | float | `per_core_M · per_core_N` tiles | somas parciais entre passos de K |

Como `c_24` é float e `A`/`B` são bfloat16, o compute reconfigura o unpacker
de SrcA antes de recarregar a soma parcial
(`copy_tile_to_dst_init_short_with_dt`) e depois volta ao formato de `B`
(`reconfig_data_format_srca`), como no exemplo 04. A recarga passa por SrcA,
que guarda TF32 (19 bits): por isso o erro é um pouco maior que o dos
exemplos 05 e 06, que nunca tiram a soma do Dst, mas fica bem abaixo da
tolerância.

## Mapeamento

| Item | Valor |
|---|---|
| Operação | `C = AB`, `m = 576`, `n = 512`, `k = 384` (`Mt = 18`, `Nt = 16`, `Kt = 12`) |
| Núcleos | 72 (grade de 8 x 9 blocos) |
| Compute | `HiFi4`, `fp32_dest_acc_en = true`; forma dos blocos como argumentos de compilação |
| Validação | `reference::gemm` em CPU, tolerância de bfloat16 |

## Saída para o benchmark

O programa executa o workload duas vezes (a primeira inclui a compilação JIT
dos kernels) e mede a segunda. Depois imprime uma linha CSV:

```
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
```

`gflops = 2mnk / tempo`. No simulador o tempo não representa o hardware.

## Como executar

```bash
./scripts/run.sh 07_gemm_v3_reuse
```

Saída esperada (o tempo varia):

```
Exemplo 07: C = A B com A 576x384, B 384x512 bfloat16 e C float em 72 de 8x9 núcleos
  bloco de 2x2 tiles por núcleo, passo de 2 tiles em K, sub-blocos de 1x2
Erro absoluto máximo: 4.487e-02 | erro relativo máximo: 1.516e-03 | PCC: 1.000000
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
07_gemm_v3_reuse,576,512,384,bf16-fp32,72,1208.462,0.187,1.516e-03
PASSOU
```

Próximo: o exemplo 08 lê cada bloco da DRAM uma só vez e o distribui por
multicast.
