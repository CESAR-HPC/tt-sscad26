# 00 — DRAM loopback

Primeiro contato com o TT-Metalium e com a **movimentação de dados**: o host
coloca dados na DRAM do dispositivo, um kernel copia-os tile a tile para
outro buffer da DRAM e o host lê o resultado de volta.

```
host --PCIe--> DRAM (entrada) --NoC--> L1 --NoC--> DRAM (saída) --PCIe--> host
```

Referência: `tt-metal/tt_metal/programming_examples/loopback`.

## O que o exemplo mostra

- **Buffers na DRAM e na L1** (`MeshBuffer`). Um núcleo Tensix não acessa a
  DRAM diretamente: todo dado passa pela L1 (SRAM local do núcleo) através
  da NoC. Por isso há um buffer de um tile na L1, usado como área de trabalho.
- **Buffers interleaved.** O buffer é dividido em páginas (aqui, 1 página =
  1 tile de 2048 B) distribuídas em round-robin entre os bancos de DRAM. O
  `TensorAccessor` converte "página i" no endereço NoC correto.
- **Transferências assíncronas.** `noc_async_read_page`/`noc_async_write_page`
  só disparam a transferência; as barreiras (`noc_async_read_barrier`,
  `noc_async_write_barrier`) esperam que ela termine.

## Mapeamento

| Item | Valor |
|---|---|
| Operação | `saída = entrada`, `num_tiles` tiles bfloat16 (padrão 8) |
| Núcleos | 1 (`{0,0}`) |
| Kernels | 1 de movimentação de dados no RISC-V 0: `kernels/dataflow/loopback.cpp` |
| CBs | nenhum |
| Validação | comparação bit a bit (uma cópia deve ser exata) |

O laço do kernel é **deliberadamente serial**: um único tile na L1 e uma
barreira após cada transferência, sem sobreposição entre leitura e escrita.
O exemplo 02 resolve isso com circular buffers e kernels de leitura e
escrita separados.

## Como executar

```bash
./scripts/run.sh 00_dram_loopback        # 8 tiles
./scripts/run.sh 00_dram_loopback 64     # 64 tiles
```

Saída esperada (além dos logs do runtime):

```
Exemplo 00: cópia DRAM -> L1 -> DRAM de 8 tiles bfloat16
Dispositivo 0 aberto: grade de núcleos Tensix 8x9 (72 núcleos)
Elementos diferentes: 0 de 8192
PASSOU
```

Próximo passo: [01_riscv_add](../01_riscv_add) — o núcleo que move os dados
também é uma CPU.
