# 风机停机预警训练平台（可续跑 + 留出样本判定）

用 FastAPI + NumPy + SQLite 实现的风机停机预警逻辑回归训练平台。训练由两个工作器
容器执行，每个批次都把权重、动量、损失和进度作为**一个原子断点**提交；工作器退出后
另一个工作器自动接管，并从最近断点继续。最终权重与不中断训练逐位一致。

提交时可附带 2～16 条**独立留出样本**（同维度、同时含正常与停机标签）以及最低召回率
和最高误报率。留出样本只用于最后一批的判定：停机概率 ≥ 0.5 记为预警，据此计算召回率
（TP/(TP+FN)）和误报率（FP/(FP+TN)）。权重、进度、两项指标和达标结论在**同一个事务**
里随完成状态一起持久化，因此查询不会看到"已完成但无结论"的任务；任何中间批次也不会
提前给出达标结论。

## 组成

| 组件 | 说明 |
| --- | --- |
| `api` | FastAPI，提交任务、查询进度/断点/权重/判定、Swagger 演示 |
| `worker-1` / `worker-2` | 同一镜像的 `python -m app.worker`，共享同一个 SQLite 文件 |
| `training-data` | Docker 卷，挂载到三个容器的 `/data` |

两个工作器共享**同一个** SQLite 数据库（`/data/training.db`），因此"检查租约并写入
断点"和"写入断点 + 判定"是同一个 `BEGIN IMMEDIATE` 事务，不需要跨库协调。

## 训练与留出样本约定

- 训练输入 8～64 条记录，每条是风机振动/电流等特征，标签 `0`=正常、`1`=停机预警；
  各特征列**只用训练样本**标准化后追加偏置列，权重长度为 `特征数 + 1`。
- 留出样本 2～16 条，特征列数与训练样本一致，标签必须**同时包含 0 和 1**（否则召回率
  或误报率无定义）。留出样本既不参与训练，也不参与列统计量的拟合；提交时用训练样本的
  均值/标准差变换后保存，判定时不再重新拟合。
- 最低召回率 `min_recall` 和最高误报率 `max_false_positive_rate` 取值 `0～1`；附带留出
  样本时缺省分别为 `0` 和 `1`（即只要求两项指标有定义）。不附留出样本时不允许给出这两个
  阈值。
- 轮数 2～5；按输入顺序每 8 条一批（最后一批可不足 8 条）。
- 逻辑回归（sigmoid + 二元交叉熵），SGD，学习率 `0.1`，动量 `0.9`，float64。
- 每批提交的断点包含：`step`、`training_epoch`、`batch_in_epoch`、`weights`、
  `momentum_velocity`、`batch_loss`、`loss`（全量交叉熵）、`progress`、提交时间、
  `worker_id`、`lease_epoch`。
- **最后一批**额外在提交事务内计算并写入：`recall`、`false_positive_rate`、
  `threshold_met`（`recall ≥ min_recall` 且 `false_positive_rate ≤ max_false_positive_rate`）
  以及完整的 `evaluation`（混淆矩阵四个计数、预测阈值 0.5、两个阈值、`evaluated` 标志）。
  未附留出样本的任务照常训练，`evaluation.evaluated=false`、`threshold_met=null`。

## 启动

```bash
docker compose up -d --build
# Swagger: http://localhost:8000/docs
```

`worker-1` / `worker-2` 故意不设 `restart` 策略，便于演示接管。被 kill 的工作器用
`docker compose start worker-1` 恢复（它会以空闲状态重新加入，不会抢占已完成任务）。

## API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/jobs` | 提交训练；可附带 `holdout_features`、`holdout_labels`、`min_recall`、`max_false_positive_rate` |
| `GET` | `/jobs` | 最近任务列表（含判定结果） |
| `GET` | `/jobs/{job_id}` | 已确认步数、进度、损失、当前权重、留出判定；完成时含 `final_weights` |
| `GET` | `/jobs/{job_id}/checkpoints` | 全部批次断点（可按 `after_step` 增量拉取） |
| `GET` | `/jobs/{job_id}/events` | 审计事件：创建、开始/从第几步续训、批失败、完成（含判定）、丢失租约 |
| `GET` | `/workers` | 工作器心跳与租约持有者 |
| `GET` | `/health` | 健康检查 |

提交示例：

```json
{
  "features": [[...], ...],
  "labels": [0, 1, ...],
  "epochs": 3,
  "holdout_features": [[...], ...],
  "holdout_labels": [1, 1, 0, 0],
  "min_recall": 0.8,
  "max_false_positive_rate": 0.2
}
```

`GET /jobs/{id}` 在完成时返回：

```json
{
  "status": "completed",
  "completed_steps": 15,
  "total_steps": 15,
  "final_weights": [1.44, 2.01, 0.17, 0.68, 0.17],
  "recall": 1.0,
  "false_positive_rate": 0.0,
  "threshold_met": true,
  "evaluation": {
    "evaluated": true, "threshold_met": true,
    "prediction_threshold": 0.5, "holdout_records": 8,
    "positive_records": 4, "negative_records": 4,
    "true_positive": 4, "false_positive": 0,
    "true_negative": 4, "false_negative": 0,
    "recall": 1.0, "false_positive_rate": 0.0,
    "min_recall": 0.8, "max_false_positive_rate": 0.2
  }
}
```

未完成时 `recall`、`false_positive_rate`、`threshold_met`、`evaluation` 均为 `null`；
未附留出样本的已完成任务 `evaluation.evaluated=false`、`threshold_met=null`。

`Swagger`（`/docs`）里直接 `POST /jobs` 提交即可看到训练开始；`GET /jobs/{id}` 与
`GET /jobs/{id}/checkpoints` 查看进度与每个断点的权重/动量/损失。

## 续跑与防回退机制

1. 工作器每批在**同一个事务**里写检查点、更新任务进度、刷新租约心跳。任何一项失败
   则整体回滚，不会出现"权重已变但进度没变"的中间态。
2. 租约保存在同一数据库的 `leases` 表。工作器每 `1.5s` 心跳；持有者心跳超过
   `6s`（`TRAINING_LEASE_SECONDS`）未更新，另一个工作器才可接管，接管时
   `epoch + 1`。
3. 每次提交前在事务内校验 `(worker_id, epoch)` 仍是租约持有者。被接管后仍存活（或
   迟到唤醒）的旧工作器校验失败，直接 `LeaseLost` 退出：既不能回退断点，也不能把
   别人的完成状态改写，更不能重复提交完成事件。
4. 完成状态由 `step == total_steps` 在提交事务内一次性置位；最后一批的召回率/误报率/
   达标结论与 `completed` 状态、断点、权重同事务写入。旧工作器无法再写入。
5. 末批计算指标或写入失败时整个事务回滚：任务仍停在最后一批之前，断点/权重/进度保持
   上一次已确认的值，不会出现半个判定。工作器随即释放租约并在 `batch_failed` 事件里记下
   原因，本工作器或另一个工作器下一轮从该断点重新计算末批直至成功；两个工作器都在时
   不会因此永久卡住。
6. 查询侧再做一致性校验：`completed` 但缺失 `evaluation_json`、或最新断点与
   `completed_steps`/权重/动量不一致时返回 500，而不是伪造结论。
7. SQLite 使用 WAL + `synchronous=FULL`，已提交的断点在进程崩溃/容器重启后仍然存在。
   旧版本创建的数据库在启动时原地补齐新列（只增可空列，不重写已有数据）。

## 演示与验证

```bash
# 完整演示（Docker）：提交 -> 观察 -> kill 当前持有者 -> 接管续跑 -> 对比不中断训练
python3 scripts/demo.py

# 没有 Docker 时用本机进程演示同样流程
python3 scripts/demo.py --local

# 不依赖 Docker 的三项自检
python3 scripts/selftest.py

# 单元/集成测试（含真实 uvicorn + 两个工作器进程、SIGKILL 接管、迟到工作器围栏）
python3 -m pytest -q
```

演示脚本第 6 步会用同一份输入（含同一留出样本）再训练一次（不中断），并断言两次的
`final_weights`、`loss`、`momentum_velocity` 以及 `recall`、`false_positive_rate`、
`threshold_met` 完全相等。

`scripts/selftest.py` 的第二项检查会在**末批之前**硬杀持有者，验证接管后：
中间查询没有判定、完成后判定只写一次、且与不中断训练完全一致。

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TRAINING_DATA_DIR` | `./data` | 数据库目录 |
| `TRAINING_DB_PATH` | `$TRAINING_DATA_DIR/training.db` | 共享数据库 |
| `TRAINING_LEASE_SECONDS` | `6` | 租约超时 |
| `TRAINING_HEARTBEAT_INTERVAL` | `1.0` | 心跳间隔（必须小于租约） |
| `TRAINING_POLL_INTERVAL` | `0.3` | 空闲轮询/批间间隔 |
| `TRAINING_BATCH_DELAY_SECONDS` | `0` | 模拟每批耗时，便于演示 |
| `TRAINING_DEMO_PAUSE_FILE` | 未设置 | 演示用：文件存在时在批间暂停（不心跳） |
| `TRAINING_DEMO_CRASH_AFTER` | 未设置 | 演示用：指定步数提交后 `os._exit` |

## 目录

```
app/main.py       API 与 Swagger
app/worker.py     工作器主循环、租约续训
app/store.py      任务状态机、批次提交、断点查询
app/lease.py      租约获取/续约/围栏校验
app/model.py      逻辑回归、交叉熵、SGD + 动量、留出指标与判定
app/features.py   输入校验、训练样本标准化、留出样本复用训练统计量
app/db.py         SQLite 连接、表结构与原地迁移
tests/            单元、判定/故障接管、进程级集成测试
scripts/demo.py   Swagger 同款接口的端到端演示
scripts/selftest.py  无 Docker 自检
```
