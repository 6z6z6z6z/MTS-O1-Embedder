O1 Embedder 是一种推理增强的稠密检索器，它模仿大型语言模型（LLMs）逐步思考的行为来解决复杂和零样本的检索任务。
它是首个将长思维链生成与判别式嵌入整合到统一框架中的检索模型，从而在领域内和分布外（OOD）的信息检索基准测试中均实现了高性能。
O1 Embedder 的训练涉及两种类型的数据。一种用于嵌入能力，由查询及其相关文档（即 q-doc 元组）构成。另一种用于思考能力，包含查询及其“思考”（即 q-thought 元组）。与广泛存在的 q-doc 元组不同，现实中并没有现成的 q-thought 元组。为解决此问题，提出了一种数据合成流程，利用 LLM 已具备的推理能力来生成此类数据集。
启动vLLM➡生成多条thought候选（默认3条）➡检索委员会投票筛选最优thought（用“多个检索器”的一致性来评估thought是否真的把query的检索意图补全了。如果某条thought让多个retriever都更容易把它和正例文档对齐，就更可能是“好thought”。）
暂时无法在飞书文档外展示此内容
- 在线时，Query 先用decoder-only LLM（也就是O1的query-side）生成 thought，把 query 的隐式推理显式化，让后续embedding更“可分”；然后把 query+thought 拼接后的序列送进embedder encoder，编码成向量 vQ。语料库 passage 预先离线编码成 vC 并建 Faiss 索引。检索时用 vQ 与 vC 做点积/余弦的 ANN 搜索拿到 Top-k，最后可选用 reranker（使用cross-encoder或LLM打分） 对 Top-k 精排得到最终排序。
- 我们如果将其应用到多维时间分类中，对训练集中的每条数据（相当于语料库）建议都生成thought再编码
- 训练（Train）= 一个 batch 前向，产生三类监督信号（让“思考能力”和“检索向量空间”共享参数，从而形成耦合）
输入：json/jsonl 样本。
包含：
  - 检索数据（q-doc）：query、pos、neg（以及可选 prompt/type/batch_size 等字段）。
  - 思考数据（q-thought）：thought（由合成流水线生成并投票筛选）。
  - 可选蒸馏：pos_scores/neg_scores（或等价的 teacher 分数）。
Dataset为每个query构造一个group（1个正例+多个负例）
Collator进行tokenize, padding, 并构造训练需要的附加字段
  - 对比学习 loss(in-batch / cross-device)（检索能力的主监督）：
    - in-batch：同一个 batch 里，别的 query 的 passage 也当作负例，形成更强的对比信号。
    - cross-device：若开启 negatives_cross_device，会跨卡 all_gather 扩大负例池。
    - 目的：让“(query + thought)”的 query 向量更靠近正例 passage，远离负例。
  - 可选 KD loss (teacher_scores)（软标签对齐）：
    - 训练数据里提供 teacher 分数（例如 pos_scores/neg_scores），或某些分支使用特定的 KD 形式时选用
    - 目的：不仅学“谁是正例”，还学“候选之间相对好坏程度”的分布。
  - 仅 O1：LM loss 生成 thought （思考能力的监督）：
    - 只对 thought 区域算：labels mask 使得模型只为 thought token 负责。
    - 目的：让模型推理时能生成更像训练分布的 thought，从而在“先想再检索”的范式下稳定工作。
  - 实际训练中通常是把多项 loss 加权求和后反传更新参数：
    - 检索对比学习是主项（L1）。
    - KD（L2）与 LM（L3）视数据/设置启用。
- 推理（Infer）= 生成多条 thought → 多视角编码 → 向量集成
  - 用 decoder-only LLM 采样多条 thought（带随机性）。n_ans 越大，集成越强，但延迟越高。
  - 对每条 thought，把“query + thought”拼起来编码，得到一条向量。直观上：每条 thought 提供一个“解释 query 的视角”。
  - 把多条向量取平均，降低采样带来的方差（self-consistency 风格的集成）。最终输出的 vQ2 才是送进 Faiss 做召回的 query embedding。