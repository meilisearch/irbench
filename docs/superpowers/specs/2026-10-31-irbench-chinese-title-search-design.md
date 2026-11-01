# IRBench — Chinese title search (C-MTEB EcomRetrieval and VideoRetrieval)

Date: 2026-10-31
Status: Draft, awaiting approval (licence and plugin rule decided 2026-11-01)
Parent specs: `2026-10-09-irbench-design.md`, `2026-10-31-irbench-amazon-esci-multilingual-design.md` (builds on its scenario language, per-language engine settings, plugin images and pinned embedding model)

## 1. Goal

Every scenario today searches documents with whole questions or with product queries in English, Japanese or Spanish. None covers two things a search box very often sees:

- **Short queries against short titles.** A shopper types `纯山楂酱` (pure hawthorn jam), a viewer types `斗罗大陆2绝世唐门2季`, and the engine has one line of text per item to match against. There is no body text to fall back on.
- **Chinese**, written with no spaces between words. Before an engine can match `宝宝腹部加棉护肚` against `护肚子肚围婴儿冬季纯棉加厚绒肚兜宝宝保暖新生儿裹腹护肚脐神器`, it has to decide where the words are, on both sides.

This change adds two scenarios built on the two Chinese title-search sets of C-MTEB, both from Alibaba's Multi-CPR:

| Scenario | Queries | What it adds |
|---|---|---|
| `ecommerce/title-search-zh` | Real e-commerce search queries (median 6 characters), against 100,902 product titles | Short Chinese queries against keyword-stuffed listing titles, a common shape for marketplace search in China. No query appears verbatim in its relevant title: matching needs segmentation and partial matching. |
| `media/video-search-zh` | Real video-site search queries (median 7 characters), against 100,930 video titles | Title search over a media catalogue: show names, episode numbers, people's names, mixed Latin and Chinese (`ukBboy`, `98shuihuzhuan`). |

Both scenarios run keyword, semantic and hybrid only (§4). They are the first scenarios with one judged document per query, so this spec also settles how such scores are reported without overstating what they show (§6).

### Non-goals

- Traditional Chinese queries or catalogues. Both sets are simplified Chinese. A simplified-to-traditional query variant is listed in §11.
- Pinyin input. A few queries are pinyin (`98shuihuzhuan`); they stay as they are, and no pinyin perturbation is generated.
- The Medical set of Multi-CPR. Its queries are questions and its passages are answers, which the knowledge-base scenarios already cover.
- Filtering, facets or sorting. The sets have an id and a title, nothing else.
- Changing the English, Japanese or Spanish scenarios.

### Anti-goals

- A scenario an engine wins only because another is set up badly. Every engine gets its documented Chinese setting on the default track, and the official or very popular Chinese plugins for it on the tuned track (§5). A reader who runs Elasticsearch in Chinese should not be able to say "nobody runs Chinese without smartcn or IK" about the tuned results.
- Publishing a ranking the data cannot support. With one judgment per query, many differences between engines are within noise. The report marks them as ties (§6) and the scenario pages say how large a difference has to be before it means anything.

## 2. Decisions

| Topic | Decision | By |
|---|---|---|
| Sets | C-MTEB EcomRetrieval and VideoRetrieval, one scenario each | user |
| Corpus | The C-MTEB corpus of each set (100,902 and 100,930 titles), exact duplicate titles merged (§3.3) | my call |
| Queries | All 1,000 C-MTEB queries, plus 3,000 Multi-CPR train queries whose judged title is in the same corpus: 4,000 per scenario (§3.2) | my call |
| Modes | keyword, semantic, hybrid. filtered, faceted, sorted are N/A: there are no attributes | my call |
| Perturbations | None (`kinds: []`). `typo` is rejected for `zh` as for `ja`; `prefix` is left out so every score is over the original queries (§4) | my call |
| Embedding model | `multilingual-e5-small`, pinned on both scenarios (§5.3) | my call |
| Headline metric | MRR@10, with success@1, recall@10, recall@100 and nDCG@10 next to it, each with its 95% interval (§6) | my call |
| Default track | Each engine's built-in Chinese setting, nothing installed (§5.1) | my call (the rule set by the multilingual spec) |
| Tuned track | Official or very popular plugins allowed (§5.2): Elasticsearch `analysis-smartcn`, `analysis-icu` and IK; Postgres PGroonga and zhparser | user (the rule); my call (which plugins meet it) |
| Licence | No licence. Runs stay internal until the Multi-CPR authors give terms (§7) | user |

## 3. Data

### 3.1 Sources

| File | Source | Pinned at | sha256 |
|---|---|---|---|
| Ecom corpus | `huggingface.co/datasets/C-MTEB/EcomRetrieval`, `data/corpus-00000-of-00001-7f0e87850b5c9454.parquet` | `687de13dc7294d6fd9be10c6945f9e8fec8166b9` | `ad50bc86e74f45ac57ad5b2f475e564a6baef4b286d0af803ac673e26ba0e059` |
| Ecom queries | same repo, `data/queries-00000-of-00001-46d6d2b2f0ad8826.parquet` | same | `ddf99eba426a21182755979e5c8f3b0afe93808096437779a7ab2d6a316df686` |
| Ecom qrels | `C-MTEB/EcomRetrieval-qrels`, `data/dev-00000-of-00001-42566aaf5580f662.parquet` | `39c90699b034ec22ac45b3abf5b0bbb5ffd421f9` | `189c199c2a3c615f8ec476549039e00f8e49f18905ede786d4003237d71d2933` |
| Video corpus | `C-MTEB/VideoRetrieval`, `data/corpus-00000-of-00001-3f9eee0a3ef9663e.parquet` | `58c2597a5943a2ba48f4668c3b90d796283c5639` | `02c3000524f7ff40c2a1d5a209f5cde9b81c7f8eadee41f6311bf2b9eeb483c0` |
| Video queries | same repo, `data/queries-00000-of-00001-718fd9848565c08f.parquet` | same | `0f3df85f0550833d7f5a009b43da777edfb9a298327ee020e02d19f47a3b1173` |
| Video qrels | `C-MTEB/VideoRetrieval-qrels`, `data/dev-00000-of-00001-06b3de608f2202bb.parquet` | `faa71382b6a29cf1778d1f436b963e75cb5b927c` | `ce205fa71e98f9d42a7c438e762484ee167ea8a38c97ab2e6f6bdfd494336e95` |
| Train queries and qrels (both sets) | `github.com/Alibaba-NLP/Multi-CPR`, `data/{ecom,video}/{train.query.txt,qrels.train.tsv}` | `a4e467182a3e2c110528a1575d79b33cc449d2c3` | the builder records them when writing `dataset.yaml` |

Measured 2026-10-31 on these files:

- The C-MTEB corpus of each set is a subset of the Multi-CPR corpus (1,002,822 ecom and 1,000,000 video passages): every C-MTEB title is in Multi-CPR with the same id and text. The C-MTEB queries and qrels are Multi-CPR's dev files, ids unchanged.
- The C-MTEB corpus is the 1,000 judged titles plus a random sample of the rest. Because the sample is random, it also holds the judged title of 9,757 ecom and 8,449 video train queries. Those queries can be scored against the C-MTEB corpus with no change to it (§3.2).
- No dev query string appears among the train queries, in either set.

### 3.2 Counts and shape

| | Ecom | Video |
|---|---|---|
| Titles | 100,902 | 100,930 (100,579 distinct texts) |
| Title length, median (p10–p90) | 31 characters (28–41); none over 200 | 24 characters (10–43); 1,402 over 200 (cast lists), longest 5,869 |
| Titles with a space | 31% | 81% |
| Titles with Latin letters / digits | 47% / 47% | 40% / 59% |
| Titles / queries with upper-case Latin | 33% / 0% | 0.8% / 3.5% |
| Titles / queries with full-width ASCII (`ＡＢＣ１２３`) | 2,496 / 7 | 119 / 24 |
| Dev queries (C-MTEB) | 1,000 | 1,000 |
| Train queries with their judged title in the corpus | 9,757 | 8,449 |
| Query length, median (p10–p90), dev and train alike | 6 characters (4–10) | 7 characters (4–12) |
| Queries with a space | 3.7% | 2.3% |
| Queries with Latin letters | 15.8% | 10.2% |
| Judgments per query | exactly 1, grade 1 | exactly 1, grade 1 |
| Dev queries found verbatim in their judged title | 0 | 12 |
| Share of a dev query's Han characters present in its judged title, median | 78% | 71% |

Examples (dev):

- ecom: `纯山楂酱` → `今明后山楂果酱山楂酱炒酸奶酱家用小包装瓶装水果果酱低脂260g`; `工作站机箱` → `撒哈拉X500/X800/X1000多硬盘位机箱支持9/12/14盘 双U服务器机箱`; `初哆咪夏红膏` → `【买2发3】夏红膏正品官方旗舰店初哆山楂膏咪药食同源`.
- video: `鸡西新闻2019` → `鸡西新闻 20191008`; `胜者为王2天下无敌正片` → `胜者为王ii天下无敌大结局`; `ukBboy` → `0359街舞俱乐部推荐视频 日本少儿kyusyu danji团队uk bboy 2012超强嘉宾表演 高清`.

What this means for an engine:
- A query is never a phrase of its title (ecom: 0 of 1,000). An engine that requires every query term, or a phrase, finds little. Partial matching decides the score.
- Ecom titles are run-on keyword lists with no spaces in 69% of them; the whole title is one "word" to an engine that only splits on spaces and punctuation.
- Case and width differ between the two sides: ecom queries are all lower case while a third of titles are not (`adidas纸袋子`, `小米WiFi放大器`), video queries keep capitals that titles drop (`ukBboy` vs `uk bboy`), and some titles use full-width letters and digits. Numbers are written as digits or Roman numerals (`2` vs `ii`).
- Traditional characters are rare on both sides (under 0.1% of titles in a spot check), so traditional/simplified folding is not what these scenarios measure (§11).

### 3.3 Loader

One normaliser, `irbench.datasets.cmteb` (`normalize_cmteb`), registered twice in `harness/irbench/datasets/__init__.py`:

- `datasets/cmteb-ecom/dataset.yaml` and `datasets/cmteb-video/dataset.yaml`, each with the five files of §3.1 and `params: {domain: ecom | video}`. `license:` says "not specified" and points at §7.
- **Documents:** `id` (the corpus id), `title` (the text, stripped). Text is otherwise kept as written: no full-width folding, no case folding, no traditional/simplified conversion. How an engine handles those is part of what is measured.
- **Duplicate titles** (my call): titles with exactly the same text are merged into the one with the smallest id, and judgments on the others are moved to it. Otherwise an engine is scored wrong for returning an identical copy of the right title, at random. The video corpus has 351 such texts (one of them a dev query's judged title); the ecom corpus has none. The manifest records how many judgments moved.
- **Queries:**
  - all 1,000 dev queries, `group: dev`, id `<set>-dev-<qid>`;
  - a seeded sample (seed `20261031`) of 3,000 train queries among those whose judged title is in the corpus, `group: train`, id `<set>-train-<qid>`;
  - text with whitespace normalised; punctuation kept (`老肉，`, `无心法师：`).
- **Qrels:** grade 1 for the one judged title. No judged non-relevant documents exist.
- The sample size and seed are loader constants, as for ESCI.

Why train queries: §6.2 shows that 1,000 queries leave intervals too wide to separate most engines. The engines are not trained, so a train query is as good a test as a dev query; only an embedding model trained on Multi-CPR would gain from it, and the chosen model was not (§5.3). The `dev` group alone gives the numbers comparable with C-MTEB, and the scenario page shows it as a group, as it shows ESCI's query sources.

## 4. Scenarios

```yaml
id: ecommerce/title-search-zh
dataset: cmteb-ecom
language: zh
embedding_model: multilingual-e5-small
headline_metric: mrr@10
index_fields:
  searchable: [title]
  filterable: []
  sortable: []
  facets: []
modes: [keyword, semantic, hybrid]
structured_bases: []
embedding_fields: [title]
filter_templates: []
sort_templates: []
top_k: 100
index_timeout_s: 1800
perturbations:
  seed: 20261031
  kinds: []
perf:
  page_sizes: [20, 100]
```

`media/video-search-zh` is the same with `dataset: cmteb-video`. It is the first scenario under `media/`; the report already groups scenarios by their directory.

- **Modes.** filtered, faceted and sorted are N/A: the data has no field to filter, facet or sort on. The scenario page says so in one line.
- **No `prefix`.** It would cut the last whitespace-separated piece to 40–70% of its characters; with 97% of queries having no space, that is the whole query (`宝宝腹部加棉护肚` → `宝宝腹部`). The report's headline, its interval, its tie test and the `dev`/`train` group rows all pool every perturbation (`perturbation: "all"`), so a prefix variant for nearly every query would make half the headline truncated queries, break the `dev` row's comparability with C-MTEB, and put the noise estimates of §6.2 on twice as many non-independent tasks. Leaving it out keeps "all" equal to the original queries without new report code. A Chinese as-you-type scenario is listed in §11. Empty `kinds` gives the original tasks only (ESCI spec §5.1); the builder confirms the harness accepts it on this branch and adds that if not.
- **No `typo`.** Chinese is typed through an input method; mistakes are homophones (the right pinyin, the wrong character), not neighbouring keys, and `str.isalpha()` is true for Han characters. Scenario validation rejects `typo` for `zh` with the same message as for `ja`.
- **Page sizes** 20 and 100, as on the other scenarios.

## 5. Engines

### 5.1 Default track: built-in Chinese setting only

The rule is the multilingual spec's (§6.1 there): each engine gets the scenario's language through the setting it documents and ships, if it has one; nothing that has to be installed. If an engine has no setting for Chinese, it runs as on any text, and its README says so. Each setting is a config key in the engine's `default` block with a `justification:` entry, and lint requires it (multilingual spec §5.2).

| Engine | Default-track setting for `zh` | What it does with `纯山楂酱` (to be confirmed by the docker tests of §9) |
|---|---|---|
| Meilisearch v1.54.3 | `localizedAttributes` with locale `cmn` on `title`, and `locales: ["cmn"]` on every search request. Without it, short Han queries can be detected as Japanese and segmented with the Japanese dictionary. | charabia: jieba segmentation, compatibility decomposition and kvariant (traditional/simplified) normalisation |
| Elasticsearch 9.5.5 | built-in `cjk` analyzer | overlapping two-character pieces (`纯山`, `山楂`, `楂酱`), width folding, lower-casing. Elasticsearch has no built-in Chinese word analyzer; the English setup's `standard` tokenizer would split into single characters. |
| Typesense 30.2 | field `locale: "zh"` | ICU-based segmentation with variant handling, per Typesense's locale guide |
| Qdrant v1.19.2 | BM25 inference option `tokenizer: multilingual` | charabia-based word segmentation. The default `word` tokenizer splits on spaces and punctuation only, so an unspaced title would be one token. |
| ParadeDB 0.26.1 | pg_search's `jieba` tokenizer on `title` (my call among its three built-in Chinese tokenizers; the docs present it as the most accurate) | jieba segmentation |
| Postgres (pgvector 0.8.7, PG17) | `simple` text search configuration | Postgres has no Chinese configuration: a run of Han characters with no space is one token, so `纯山楂酱` only matches a title containing exactly the token `纯山楂酱`. Keyword scores near zero are the expected, shipped behaviour. |

The builder confirms each setting's name and behaviour on the pinned version (for ParadeDB, the tokenizer names changed between pg_search releases; for Qdrant, that `multilingual` is accepted as a BM25 inference option and segments Chinese, not just Japanese). If a setting does not exist as described, the engine runs without it and its README says why.

### 5.2 Tuned track: official or very popular plugins allowed

The tuned track is found as for the other scenarios: a sweep of documented search settings, scored on nDCG@10 for each base (parity doc), plus the Chinese plugins below. A plugin is tried, not imposed: it is kept only if it wins, and its best score is reported either way (multilingual spec §6.2).

**Which plugins (user, 2026-11-01): official or very popular.** A plugin qualifies if it is published by the engine's vendor, or if it is a very popular community plugin for that language on that engine. "Very popular" means all of:
- it is the plugin that guides and the engine's own community most commonly point to for that language on that engine (the justification cites the evidence: adoption figures, managed services that ship it, vendor docs that mention it);
- it has an open-source licence;
- its maintainer publishes a release or tag that builds against the pinned engine version;
- it is installed unmodified, pinned by URL and sha256 (a release file) or by commit (a source build), in the engine's zh image.

Not allowed: forks, plugins written or patched for the benchmark, and custom dictionaries. Every plugin uses the dictionary it ships with.

| Engine | Sweep adds |
|---|---|
| Elasticsearch | Official: `analysis-smartcn` (Elastic's Chinese word segmenter) and `analysis-icu` (`icu_tokenizer`, dictionary-based word breaking for Chinese), version-locked to 9.5.5. Very popular: IK (`infinilabs/analysis-ik`, Apache-2.0, about 17.5k GitHub stars), from its maintainer's 9.5.5 release zip pinned by URL and sha256. All three in one image, `compose.zh.yaml`, built like the kuromoji image. Variants: `standard`, `cjk`, `smartcn`, `icu_tokenizer`, `ik_smart`, `ik_max_word`. `standard` is in the list on purpose: on the ecom toy runs of §6.2, one token per character beat two-character pieces by 0.105 MRR@10, so the default track's `cjk` is not assumed to be better. |
| Postgres | Official package: PGroonga, as in the multilingual plan (default `TokenBigram`, which splits Han text into bigrams; the sweep also tries `TokenNgram`). Very popular: zhparser (`amutu/zhparser`, PostgreSQL-style licence, on the SCWS segmenter), the usual Chinese parser for Postgres text search. It has no PGDG package, so the zh image builds SCWS and zhparser from pinned upstream commits, unmodified, with their shipped dictionary, and the text search configuration maps nouns, verbs, adjectives and the other content token types to `simple`, as zhparser's README shows. Variants: `simple` (default), PGroonga `TokenBigram`, PGroonga `TokenNgram`, zhparser. One image `compose.zh.yaml` holds both extensions, built on the PGroonga image if the multilingual plan produced one, otherwise on the current pgvector image with zhparser only. If zhparser does not build against PG17 from an upstream commit, it is left out and the README says so. |
| ParadeDB | its other built-in tokenizers, `chinese_lindera` and `chinese_compatible` (one character per token) |
| Meilisearch, Typesense, Qdrant | no plugin exists; usual sweep of search settings (Meilisearch matching strategy, Typesense `drop_tokens_threshold`, Qdrant BM25 `k`/`b`, and so on) |

One rule is added for these two scenarios (my call): because the judgments are sparse, a sweep variant replaces the default only if its gain is significant. Otherwise a sweep would often keep a variant that won by noise. Precisely:
- Each setting is swept as today. Every variant of a setting is compared with the **default-track value** of that setting (not with the best so far), by the paired randomization test on per-query nDCG@10.
- The p-values of one setting's variants are Holm-corrected together (four Elasticsearch analyzers: three comparisons). Variants with corrected p < 0.05 and a higher score are eligible; the eligible one with the highest score is kept. If none is eligible, the default value stays.
- So if `standard` beats `cjk` significantly and `smartcn` beats `standard` but not significantly, the variant with the higher score among those significantly above `cjk` is kept: `smartcn` if it is itself significantly above `cjk`, otherwise `standard`.
- Index-time settings (analyzer, tokenizer, PGroonga tokenizer) serve both keyword and hybrid from one index. They are chosen on the **keyword** base, where they act alone; hybrid then sweeps only its own search-time settings (fusion weights) on top.
- The justification records the gain, the corrected p and the variants tried.

Tokenizer and plugin variants change the index, so each re-indexes 100k short titles; at this size that is minutes per variant, so the sweep stays cheap (an estimate; the builder records the times).

### 5.3 Embedding model: `multilingual-e5-small`

- Same model as the Japanese and Spanish scenarios: 384 dimensions (index memory comparable with every other scenario), MIT licence, `query: ` and `passage: ` prefixes, registered by the multilingual plan with `doc_prefix`. No new model code.
- It was not trained on Multi-CPR. MTEB's model metadata lists Multi-CPR among the fine-tuning data of the BGE Chinese models (from the C-Pack paper), so `bge-small-zh-v1.5` would have seen 100,000 in-domain training pairs per set, including the train queries of §3.2. Semantic and hybrid scores would then reflect that training as much as the engines.
- Exact (brute-force) cosine retrieval with each candidate, measured 2026-10-31 on the 1,000 dev queries and the C-MTEB corpus, with each model's documented prefixes, next to the toy BM25 of §6.2:

  | | Ecom MRR@10 | Ecom success@1 | Ecom recall@100 | Video MRR@10 | Video success@1 | Video recall@100 |
  |---|---|---|---|---|---|---|
  | multilingual-e5-small | 0.487 | 0.394 | 0.857 | 0.531 | 0.429 | 0.901 |
  | bge-small-zh-v1.5 | 0.521 | 0.415 | 0.905 | 0.616 | 0.508 | 0.951 |
  | toy BM25, one token per character | 0.494 | 0.398 | 0.877 | 0.552 | 0.458 | 0.892 |

- bge-small-zh is ahead by 0.034 (ecom) and 0.085 (video) MRR@10. Some of that is being a Chinese model, some is having trained on these sets' train split; the two cannot be separated here. With multilingual-e5-small, the semantic side is level with a plain keyword baseline, so hybrid has two comparable halves to fuse and neither side dominates by construction.
- The benchmark compares engines on the same vectors, so the model only needs to be a reasonable, untainted one. The semantic columns are read across engines, never against the English scenarios. If the user prefers the stronger Chinese model despite its in-domain training, the change is the `embedding_model` line and a 512-dimension vector size; nothing else in this spec depends on it.
- `text-embedding-3-small` is not run on these scenarios, as for Japanese and Spanish.

## 6. Scoring with one judgment per query

### 6.1 What one judgment means

Each query has one title marked relevant, and nothing marked irrelevant. So:
- every metric reduces to the rank of that one title: MRR@10 is 1/rank (0 past 10), success@1 is "ranked first", recall@k is "in the top k", nDCG@10 is 1/log2(rank+1);
- every other title counts as wrong, even when it is the same product from another seller or another episode of the same show. In the ecom catalogue, many listings sell the same thing. Scores are lower bounds, and an engine that ranks an equally good title first loses as much as one that ranks a bad one first;
- judged@10 is at most 0.1 by construction, so it says nothing here.

That last point is the main risk to the comparison: the false negatives are not necessarily spread evenly. An engine that matches more literally may sit closer to the one listing the annotator saw. The scenario pages say so, and §11 lists a way to measure it.

### 6.2 How noisy, measured

Two toy BM25 runs in Python give the scale. They are not engines, only two plausible ways to tokenise: one token per Han character (`unigram`, what Elasticsearch's `standard` tokenizer does), and overlapping two-character pieces (`bigram`, what `cjk` does), with Latin and digit runs as words, k1 1.2, b 0.75. Measured 2026-10-31 on the 1,000 dev queries and the C-MTEB corpus, 95% bootstrap intervals:

| | Ecom unigram | Ecom bigram | Video unigram | Video bigram |
|---|---|---|---|---|
| MRR@10 | 0.494 ±0.028 | 0.388 ±0.026 | 0.552 ±0.027 | 0.543 ±0.027 |
| success@1 | 0.398 ±0.031 | 0.300 ±0.028 | 0.458 ±0.032 | 0.446 ±0.031 |
| recall@10 | 0.695 ±0.029 | 0.583 ±0.030 | 0.746 ±0.026 | 0.742 ±0.027 |
| recall@100 | 0.877 ±0.021 | 0.803 ±0.025 | 0.892 ±0.020 | 0.905 ±0.019 |
| nDCG@10 | 0.542 ±0.027 | 0.435 ±0.026 | 0.599 ±0.025 | 0.591 ±0.025 |

Paired difference, bigram minus unigram (randomization test, 10,000 rounds):

| | Ecom | Video |
|---|---|---|
| MRR@10 | −0.105 [−0.128, −0.083], p = 0.0001 | −0.009 [−0.029, +0.013], p = 0.42 |
| success@1 | −0.098, p = 0.0001 | −0.012, p = 0.44 |
| nDCG@10 | −0.107, p = 0.0001 | −0.007, p = 0.47 |
| Queries whose rank changed | 453 | 386 |

What this says:
- **At 1,000 queries, one engine's score is known to about ±0.027 (MRR@10) and ±0.03 (success@1).** Two runs on the same queries are compared more finely: the paired interval on a difference is about ±0.02. Differences under about 0.02–0.03 are not distinguishable from noise; on video, two tokenisations that change the rank of 386 of 1,000 queries end up 0.009 apart with p = 0.42.
- **At 4,000 queries** (§3.3) the intervals halve (√4): about ±0.013 for a run, ±0.01 for a paired difference. This is the reason for adding train queries. It is an estimate scaled from the dev measurement; the builder re-measures it on the real runs and the docs page gives the measured value.
- **MRR@10 and nDCG@10 move together**: every difference above has the same sign and nearly the same size on both. Sweeping on nDCG@10 and headlining MRR@10 do not pull in different directions.
- **Absolute scores are an artefact of the corpus**: the Multi-CPR paper reports BM25 at 0.225 MRR@10 on the full 1M ecom corpus; on C-MTEB's 100k subset this toy gets 0.494. Only comparisons between engines on this corpus mean anything.
- **False negatives are visible at once.** Among the toy's wrong first results: `诊所输液椅子` (clinic infusion chair) → `输液椅子沙发单人位医院输液椅卫生院诊所用候诊椅座椅电动医用`, a clinic infusion chair, scored wrong because the annotator judged another listing; `老公们的私房钱第十集` → an episode clip of that show, scored wrong because the judged title is the show's catalogue entry. Top-10 results carry a judgment 6.6% (ecom) and 8.3% (video) of the time.

### 6.3 Report changes

- **`Scenario.headline_metric`**: `ndcg@10` (default; every existing scenario) or `mrr@10`. The relevance matrix on the scenario page, its 95% interval and the tie marking (≈, paired randomization test against the column's best) use it. The matrix heading names the metric.
  - Why MRR@10 here (my call): it is the metric of the source paper, it reads directly ("the right title is on average around rank 1/x"), and with a single grade-1 judgment nDCG@10 is the same information on a squashed scale. nDCG@10 stays in the table next to it.
  - The sweeps stay on nDCG@10 (parity doc unchanged): with one judged title per query the two rank variants almost identically (§6.2 checks it on the toy runs).
- **The report reads run bundles, not scenario files**, so the score stage writes what the report needs into `scores.json`: `headline_metric`, and `single_judgment: true` when every query has exactly one judged document.
- **Per-task scores and intervals for more metrics.** `scores.json` today keeps per-task nDCG@10 and its interval only. It gains per-task values for the headline metric, and a 95% interval for each of nDCG@10, MRR@10, success@1, recall@10 and recall@100 (`ci95: {metric: [lo, hi]}`, keeping `ndcg_ci95` so older bundles load). Normal approximation, as today; at n = 4,000 it is accurate for these means.
- **A second table** on these scenario pages: success@1, recall@10, recall@100 and nDCG@10 per run and base, each with ±.
- **judged@10** is left out of a scenario's page when its bundles say `single_judgment`, replaced by one sentence saying why.
- **Groups.** The `dev` / `train` groups appear in the existing per-group table. The `dev` row is the one comparable with published C-MTEB numbers.
- **Significance.** What the page publishes as "better" is only what the tie test supports: a run not significantly below the best is marked ≈, as today. The scenario's docs page states the smallest difference §6.2 found detectable. No correction for multiple comparisons is added; the existing pages have none, and the ≈ mark already errs toward ties.

## 7. Licence and publication

- Multi-CPR (`github.com/Alibaba-NLP/Multi-CPR`) has no licence file and its README states no terms. The C-MTEB copies on Hugging Face have no licence either, and MTEB records the tasks as "not specified". Without a licence, the default is that the authors keep all rights.
- What IRBench redistributes: nothing of the data. Raw files are downloaded at prepare time and never committed; published run bundles (`manifest.json`, `scores.json`, `perf.json`, `index.json`) hold task and document ids and numbers, no text. Rendered requests, which contain query text, are not published.
- What publishing would show: scores per engine, as the MTEB leaderboard does publicly for these two tasks. The scenario docs page would show no query or title text beyond what the paper itself prints.
- **Decision (user, 2026-11-01): these scenarios stay internal until the Multi-CPR authors give terms.** `Scenario.publishable: bool = True`, set to `false` on both, and `irbench publish` refuses a run of a non-publishable scenario with a message pointing here. Local reports (`irbench report --results .irbench/runs`) work as usual. The scenario docs pages say the results are internal and why.
- If the authors grant terms that allow publishing scores, flipping `publishable` on both scenarios is the whole change; the published bundles still carry no text.

## 8. Harness changes

Most of what these scenarios need comes from the multilingual plan (`docs/superpowers/plans/2026-10-31-irbench-amazon-esci-multilingual.md`): engine language keys in config `default` blocks and their lint check, `TrackConfig.compose`, `multilingual-e5-small` with `doc_prefix`, the PGroonga image. This plan is built after those tasks have landed. New here:

- `Scenario.language` gains `zh`; validation rejects `typo` for `zh`.
- `Scenario.headline_metric` and `Scenario.publishable` (§6.3, §7).
- Score stage: per-task headline values, intervals for five metrics, `headline_metric` and `single_judgment` in `scores.json` (§6.3).
- Report: headline metric in the matrix and tie test; the second metrics table; judged@10 hidden for single-judgment scenarios.
- `irbench publish` refuses non-publishable scenarios.
- Loader `irbench.datasets.cmteb` and two `dataset.yaml` files. The train files are plain TSV; the parquet files are read with pyarrow. No new dependency.
- The Elasticsearch `compose.zh.yaml` and its Dockerfile (smartcn, icu, IK); the Postgres `compose.zh.yaml` and its Dockerfile (PGroonga if available, zhparser). Both pin every download by URL and sha256 or commit; the `image-pin` lint rule covers the Dockerfiles as in the multilingual plan.
- `docs/methodology/parity.mdx`: the tuned-track plugin rule of §5.2 (official or very popular, with its four conditions), worded as a general rule.

## 9. Testing

- **Loader** (small fixture files): ids prefixed by set and split; 1,000 dev plus a train sample stable for a seed; only train queries whose judged title is in the corpus are eligible; duplicate titles merged into the smallest id and their judgments moved; text not folded; whitespace normalised, punctuation kept.
- **Scenario:** `zh` accepted, `typo` with `zh` rejected; `headline_metric` default `ndcg@10`; `publishable` default true.
- **Score:** per-task headline values and the five intervals on a fixture run; `headline_metric` and `single_judgment` written to `scores.json`; a bundle without them still loads.
- **Sweep rule:** a helper that takes per-query scores for a default and its variants returns the kept variant per §5.2 (Holm-corrected; default kept when none is significant; the worked case of §5.2 gives `standard` or `smartcn` as stated).
- **Report:** a scenario with `headline_metric: mrr@10` marks the best and ties on MRR@10; existing scenarios render unchanged; judged@10 hidden when every query has one judgment.
- **Publish:** a run of a non-publishable scenario is refused.
- **Adapters:** for each of the 6, the rendered index body or request for `zh` contains the setting of §5.1.
- **Docker tests** (marked `docker`), one per engine: index `护肚子肚围婴儿冬季纯棉加厚绒肚兜宝宝保暖新生儿裹腹护肚脐神器` and check that the query `宝宝护肚` finds it with the default-track setting (Postgres excepted, where the test asserts it does not, documenting §5.1). Elasticsearch's zh image: `_analyze` with `smartcn`, `icu_tokenizer`, `ik_smart` and `ik_max_word` each returns more than one token for `纯山楂酱`. Postgres's zh image: `to_tsvector` with the zhparser configuration returns more than one lexeme for `纯山楂酱`, and the `宝宝护肚` query finds the title above.

## 10. Runs

On the laptop (`local` profile), after the multilingual plan's runs:

1. Prepare both datasets; embed 2 × 100k titles and 2 × 4,000 queries with multilingual-e5-small.
2. Default track, all 6 engines, both scenarios.
3. Sweeps with the significance rule of §5.2, then the tuned track.
4. Report built locally. Nothing is published (§7).

Risks:
- **False negatives favour one kind of engine** (§6.1). Not measured by this change; §11.
- **Qdrant's `multilingual` tokenizer may not segment Chinese** as a BM25 inference option on v1.19.2. If not, Qdrant runs with `word` and its README says so; that is a result.
- **Postgres keyword near zero on the default track.** Expected (§5.1). Hybrid then rests on the semantic side.
- **Long video titles.** One title has 5,869 characters; e5-small truncates at 512 tokens, as it does for every engine.

## 11. Later

- A false-negative audit: for 200 queries, label each engine's top 3 by hand (or with an LLM judge checked against a hand-labelled sample), to measure how often an "error" is an equally good title and whether that differs by engine.
- Chinese as-you-type: prefix variants in their own scenario on the same dataset (as `ecommerce/as-you-type` does for ESCI), so they never mix into these headlines.
- A simplified-to-traditional query variant (OpenCC), to test what Meilisearch's kvariant normalisation and Typesense's variant handling claim.
- The full Multi-CPR corpus (1M titles per set) for a large-catalogue variant.
- Pooling both scenarios' queries for a single "Chinese" significance test, for twice the queries.

## 12. Questions answered by the user

- Licence: keep internal until the Multi-CPR authors give terms (§7).
- Tuned-track plugins: official or very popular (§5.2). Applying the rule, IK and zhparser qualify (my reading; zhparser needs a source build from pinned upstream commits, which the rule's fourth condition allows).
- Not changed by this spec: the Japanese and Spanish tuned tracks were specified under "official only". Whether a very popular Japanese plugin should now be tried there (for example Sudachi for Elasticsearch) is for that spec and its plan.

Calls made in this spec, to check while reading:
- 4,000 queries per scenario, 1,000 dev and 3,000 train (§3.3);
- duplicate titles merged (§3.3);
- no perturbations: no `typo`, and no `prefix` so headlines are over original queries (§4);
- ParadeDB's `jieba` as its default-track tokenizer (§5.1);
- the four conditions that define "very popular", and IK and zhparser meeting them (§5.2);
- the sweep's significance rule: each variant against the default, Holm-corrected, index-time settings chosen on keyword (§5.2);
- multilingual-e5-small over bge-small-zh (§5.3);
- MRR@10 as the headline, sweeps still on nDCG@10 (§6.3).
