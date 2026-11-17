# URS 五个微调专家的 TIES-Merging 实验

2026-11-17。使用用户已完成的CVRP、CVRPB、CVRPTW、OCVRP、OCVRPTW五个epoch30专家，以及它们共同的URS多任务base。只进行参数合并与原数据评测，不重新训练，不增加任务或数据划分。

## 实现与固定主结果

官方来源 https://github.com/prateeky2806/ties-merging ，固定commit `44e7891fc84f3de7e4caa52664cd864ca3715e91`。逐字提取官方 `src/ties_minimal.ipynb` 的参数排序/向量化、全局trim、mass sign election、zero-sign majority和disjoint mean函数。唯一适配为读取URS的model_state_dict和保存原模型可加载的checkpoint。保留BSD许可证。旧merge_lab/merge.py的简化TIES不用于本轮。

先报告官方免调参设置K=20%、lambda=1。五个专家是合并输入，base只作共同参考，不作为第六个专家参与投票：theta_merge=theta_base+lambda*TIES(theta_i-theta_base)。每个输出是一份固定权重，五任务共享该模型与同一超参数。

## 预先固定的配置

- TIES主网格：K={10,20,30,50,70,90}%；lambda={0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0,2.5}。
- K=20%额外完整覆盖官方README的lambda=0.8到2.5、步长0.1。合计66个TIES配置。
- 对照：五专家等权平均，以及Task Arithmetic的lambda={0.1,0.4,0.6,0.8,1.0}。TA lambda=0.2数学上等于等权平均，复用平均作为该点，不重复评测。
- 共72个新checkpoint，每个完整评测五项任务；配置在configs.json中于首个合并结果评测前确定。
- 原URS base和五个单专家的既有6×5结果作参考，先核验权重和协议。对应任务专家的五项对角值只作为五模型参考，不当成一个多任务模型。

## 原评测协议和选择规则

原URS DataFinder/DataReader/Tester、原dataset前1000例、N100、seed3407、greedy、POMO多起点、对称8倍增强、batch125。任务顺序固定cvrp,cvrpb,cvrptw,ocvrp,ocvrptw。保持CVRPB原80起点，其他四项100起点。

使用2×4090和2×3090并行扫描；首先在两种卡上复测base与默认TIES，要求gap差异不超过0.0001个百分点，否则正式扫描仅用4090。最终默认TIES、最佳平均gap TIES、最佳TA和等权平均在4090复核，最后报告采用4090结果。每个模型的GPU、原始JSON与耗时都记录。

主要指标为五任务aug_gap等权平均（越低越好），同时报告每任务gap、相对base的百分点差、改善任务数及最差任务退化。扫描选择按五任务平均gap最小，平分按配置ID排序；禁止按任务切换合并参数。因用户指定不新增split，调参最优值标为同一数据上的探索性最优，免调参默认值单独列出。

## 必要核查与输出

函数AST与官方一致；真实checkpoint向量化/还原恒等、原Model strict-load、所有输出权重有限；输入checkpoint SHA与此前最终核验记录一致。对官方的阈值并列/zero-sign语义做一次确定样例对照，不建立大型测试框架。

输出包含：来源和输入哈希、72个模型、全部原始评测JSON、完整CSV、默认/最优比较表、参数热图、最佳固定模型路径和核验记录。原论文URS数字与本地base分列保留。扫描结果本身不构成独立数据上的泛化证据。
