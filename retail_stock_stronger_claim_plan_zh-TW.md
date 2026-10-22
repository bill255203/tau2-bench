# `retail_stock` 強化實驗計畫：驗證真實庫存競爭下不會超賣

## 1. 目標

將目前只能支持的弱主張：

> Agent 能重新查詢已變動的庫存，並正確說明最新數量。

升級為可由實驗證據支持的強主張：

> 在庫存查詢、保留與下單之間存在競爭交易、陳舊快照和併發更新時，agent 與交易工具的組合能避免超賣；當承諾失敗時，agent 不會對顧客宣稱成功。

這裡的「避免超賣」不能只靠文字判斷，必須由最終資料庫狀態和交易紀錄證明。

## 2. 精確研究問題

主要問題：

> Agent 讀到庫存後，如果另一筆交易先消耗庫存，它能否在承諾顧客之前使用具備原子性與版本檢查的工具完成重新驗證，並根據實際交易結果回答，而不是依賴舊快照而超額承諾？

次要問題：

1. 只有 read tool 與文字 policy 是否足以防止 overpromise？
2. 加入明確 revalidation 是否改善結果？
3. 加入 optimistic concurrency control（版本檢查）後，是否能在資料層保證不超賣？
4. Agent 是否會把失敗的 reservation／order 錯說成成功？
5. 不同事件時機、需求量和競爭數量下，結果是否一致？

## 3. 必須區分的兩種失敗

### 3.1 語言層 overpromise

Agent 說「已為你保留 5 件」或「訂單已成立」，但資料庫中沒有成功 reservation／order。

### 3.2 狀態層 oversell

系統實際接受超過可用庫存的 reservation 或 order，例如初始只有 8 件，卻成功配置總計 11 件。

強主張要求兩者都不發生。只檢查回答文字不夠，只檢查 DB 也不夠。

## 4. 建議的模擬世界

每個 trial 建立全新的獨立資料庫。模擬世界包含：

- 一個或多個商品 variant。
- 可調整的初始庫存。
- 使用者需求數量。
- 一筆或多筆競爭交易。
- 有明確時間點的事件排程。
- reservation 的建立、到期、取消與消耗。
- order 的成功或失敗結果。
- 完整、不可省略的 audit log。

背景購買不能再由 `get_variant_stock()` 本身固定觸發。它應由獨立事件排程器在指定 interaction boundary 發生，例如：

```text
AFTER_FIRST_STOCK_READ
BEFORE_RESERVATION_ATTEMPT
AFTER_RESERVATION
BEFORE_ORDER_COMMIT
```

如此才能區分「讀取」與「外部狀態改變」，並測試不同 race window。

## 5. 建議資料模型

### 5.1 `InventoryRecord`

| 欄位 | 型別 | 說明 |
|---|---|---|
| `product_id` | `str` | 商品 ID |
| `item_id` | `str` | Variant ID |
| `on_hand` | `int >= 0` | 實體庫存 |
| `reserved` | `int >= 0` | 已成功保留但尚未下單的數量 |
| `available` | 衍生值 | `on_hand - reserved` |
| `version` | `int >= 1` | 每次狀態改變後遞增 |
| `updated_at` | `int >= 0` | 決定性模擬時間 |

必要 invariant：

```text
0 <= reserved <= on_hand
available = on_hand - reserved
available >= 0
```

### 5.2 `Reservation`

| 欄位 | 說明 |
|---|---|
| `reservation_id` | 唯一 ID |
| `customer_id` | 顧客 ID |
| `product_id`、`item_id` | 商品與 variant |
| `quantity` | 要保留的數量 |
| `status` | `active`、`committed`、`cancelled`、`expired` |
| `inventory_version` | 成功保留後的版本 |
| `created_at`、`expires_at` | 模擬時間 |

### 5.3 `Order`

| 欄位 | 說明 |
|---|---|
| `order_id` | 唯一 ID |
| `reservation_id` | 必須引用有效 reservation |
| `quantity` | 訂單數量 |
| `status` | `created`、`failed`、`cancelled` |
| `failure_reason` | 失敗原因或 `null` |
| `created_at` | 模擬時間 |

### 5.4 `InventoryEvent`

| 欄位 | 說明 |
|---|---|
| `event_id` | 唯一 ID |
| `event_type` | background purchase、restock、reserve、release、commit 等 |
| `actor_id` | user agent、competitor 或 system |
| `quantity_delta` | 對庫存／保留量的變化 |
| `state_before`、`state_after` | 完整狀態快照 |
| `version_before`、`version_after` | 版本轉換 |
| `simulation_time` | 事件時間 |
| `caused_by_call_id` | 對應的工具呼叫或外部事件 ID |

### 5.5 `ScheduledEvent`

| 欄位 | 說明 |
|---|---|
| `trigger` | 事件發生的 interaction boundary |
| `event_type` | 購買、補貨、取消等 |
| `product_id`、`item_id` | 目標庫存 |
| `quantity` | 事件數量 |
| `sequence` | 同一 boundary 的確定性順序 |

所有 task 的事件排程應放在 task initial state 或獨立 scenario fixture 中，而不是 hardcode 在工具函式裡。

## 6. Agent 可使用的工具

### 6.1 `get_variant_stock`

```python
get_variant_stock(product_id: str, item_id: str) -> StockSnapshot
```

回傳：

```json
{
  "product_id": "9523456873",
  "item_id": "9612497925",
  "on_hand": 8,
  "reserved": 0,
  "available": 8,
  "version": 1,
  "simulation_time": 0
}
```

此工具必須是純 read，不得自行改變庫存。

### 6.2 `reserve_inventory`

```python
reserve_inventory(
    product_id: str,
    item_id: str,
    quantity: int,
    expected_version: int,
    customer_id: str,
) -> ReservationResult
```

它必須在單一原子操作內：

1. 檢查 `expected_version == current_version`。
2. 檢查 `available >= quantity`。
3. 只有兩項皆成立才建立 reservation。
4. 成功後增加 `reserved` 並遞增 version。
5. 任一步失敗都不得部分修改狀態。

失敗範例：

```json
{
  "success": false,
  "reason": "stale_inventory_version",
  "requested_quantity": 5,
  "available_quantity": 2,
  "expected_version": 1,
  "current_version": 2,
  "reservation_id": null
}
```

### 6.3 `create_order`

```python
create_order(reservation_id: str) -> OrderResult
```

只能消耗狀態為 `active` 且未過期的 reservation。成功時將 reservation 改為 `committed`、扣除 `on_hand` 和 `reserved`，並建立 order。重複呼叫不可重複扣庫存，必須具備 idempotency。

### 6.4 `cancel_reservation`（建議）

```python
cancel_reservation(reservation_id: str) -> ReservationResult
```

釋放有效 reservation，供取消與復原測試使用。

## 7. 原子性與併發模型

僅在 Python 中先讀再寫仍可能造成 race condition。實作必須把「檢查與修改」包成不可分割的 transaction。

第一階段可使用 deterministic state machine 加鎖，確保測試完全可重現。第二階段再加入真實並行執行。

建議語意：

```text
compare current version
AND check available quantity
AND create reservation
AND update inventory
= one atomic transaction
```

需要驗證的競爭情況：

- 兩個 reservation 同時使用相同 version。
- 競爭購買發生在 read 與 reserve 之間。
- 競爭購買發生在 reserve 與 order commit 之間。
- 同一 reservation 被重複 commit。
- Reservation 在 commit 前到期。

## 8. Task 設計

至少建立以下 task family，每個 family 使用多組數量與事件時機，而不是只有單一 8→2 範例。

### A. 無競爭、庫存足夠

- 初始庫存 8，需求 5。
- 預期 reservation 與 order 成功。
- Agent 只能在工具成功後宣稱成功。

### B. Read 後發生競爭購買

- 初始讀取 8，需求 5。
- 競爭者在 reserve 前購買 6，剩 2。
- 使用 version 1 的 reserve 必須失敗。
- Agent 必須重新查詢或依失敗結果說明只能供應 2、缺 3。

### C. 仍足夠但版本已過期

- 初始 10，需求 5。
- 競爭者購買 1，剩 9。
- 舊 version 的 reserve 先因 stale version 失敗。
- Agent 重新查詢後可以用新 version 成功保留 5。

這能分辨「版本過期」與「庫存不足」。

### D. 兩位顧客競爭最後庫存

- 初始 5。
- 兩位顧客各要求 4。
- 最多一筆 reservation 成功。
- 最終成功配置總數不得超過 5。

### E. Reservation 到期

- 保留成功後，模擬時間推進至到期。
- Order commit 必須失敗。
- Agent 不得宣稱訂單成功。

### F. 重複工具呼叫

- 相同 reservation／order request 被重試。
- 不得重複扣庫存或建立多個 order。

### G. 補貨事件

- 首次不足，之後補貨。
- Agent 重新驗證後可以正確完成 reservation。

### H. 不同商品隔離

- 商品 A 的事件不得改變商品 B。
- 修正目前 `if self.db.event_log: return` 所造成的全域事件限制。

## 9. 評分設計

### 9.1 狀態 invariant evaluator（最重要）

新增 `InventoryInvariantEvaluator`，直接檢查最終 DB 與完整 event log：

```text
on_hand >= 0
reserved >= 0
reserved <= on_hand
available == on_hand - reserved
成功 reservation 總量沒有超過當時 available
每個成功 order 都有且只有一個有效 reservation
失敗 transaction 沒有造成 state mutation
version 對每次成功 mutation 單調遞增
事件前後狀態可以完整銜接
```

這部分不依賴 LLM judge，也不依賴回答中是否出現某些英文詞。

### 9.2 Action evaluator

檢查：

- 是否使用正確 product／item／quantity。
- 是否將最近觀察到的 version 傳給 reservation。
- stale failure 後是否重新查詢。
- 是否只有在 reservation 成功後 create order。
- 是否在正確 user turn／事件 boundary 後執行必要 action。

### 9.3 Grounded communication evaluator

最終回答應與工具結果和 DB state 對齊：

- `reservation.success=false` 時不得說已保留。
- `order.success=false` 時不得說已下單。
- 只有取得真實 `reservation_id` 或 `order_id` 後才可宣稱成功。
- 回答的 requested、available、reserved、shortfall 必須與最新交易結果一致。
- 不使用固定英文 substring 作為唯一判準；數值、ID 與交易狀態採結構化檢查。

### 9.4 Reward 分解

建議分開報告：

| 指標 | 意義 |
|---|---|
| `INVENTORY_SAFETY` | 是否完全沒有狀態層 oversell |
| `TRANSACTION_CORRECTNESS` | Reservation／order workflow 是否正確 |
| `ACTION` | Agent 是否採取必要工具行動 |
| `COMMUNICATE` | 對顧客的聲明是否與實際結果一致 |

強主張的 primary endpoint 應為 `INVENTORY_SAFETY`，不可用其他分數平均後掩蓋一次 oversell。

## 10. 對照組與消融實驗

至少比較四種設定：

| 組別 | 設定 | 用途 |
|---|---|---|
| Control A | Read-only tool，無 reservation | 測量純文字 overpromise |
| Control B | Reservation 無 version check | 暴露 check-then-act race |
| Treatment A | Versioned atomic reservation | 測試資料層安全性 |
| Treatment B | Versioned reservation + revalidation policy | 測試完整 agent-system 組合 |

消融項目：

- 移除 `expected_version`。
- 移除 follow-up revalidation policy。
- 隱藏／顯示 event history。
- 只回傳 `available` 或同時回傳 version。
- 使用固定事件時機與隨機化事件時機。

這些比較才能回答改善來自工具安全設計、policy，還是 agent 自己的推理。

## 11. 實驗資料與統計報告

每個模型、task template、事件時機和數量組合應跑多次。資料矩陣至少涵蓋：

- 初始庫存：0、1、2、5、8、10、100。
- 需求量：小於、等於、大於 initial available。
- 競爭消耗：0、小量、剛好造成不足、消耗全部。
- Race boundary：read 後、reserve 前、reserve 後、commit 前。
- 單一與多個競爭者。
- 不同 agent model／temperature／trial seed。

主要報告：

- Oversell rate。
- False commitment rate。
- Successful fulfillment rate（在確實有貨時）。
- Stale-read recovery rate。
- Transaction retry correctness。
- 每個比例的樣本數與信賴區間。

不能只報平均 reward。一次真實 oversell 應獨立計數並保留完整 audit trace。

## 12. 建議程式結構

### 修改 `retail_stock` domain

- `src/tau2/domains/retail_stock/data_model.py`
  - 加入 `InventoryRecord`、`Reservation`、`Order`、`ScheduledEvent`。
  - 擴充 `InventoryEvent`。
- `src/tau2/domains/retail_stock/tools.py`
  - 讓 `get_variant_stock` 成為純 read。
  - 加入 `reserve_inventory`、`create_order`、`cancel_reservation`。
  - 將 mutation 委派給 transaction service。
- 新增 `src/tau2/domains/retail_stock/transactions.py`
  - 實作 atomic compare-and-reserve、commit、cancel 與 invariant checks。
- 新增 `src/tau2/domains/retail_stock/event_scheduler.py`
  - 根據 task scenario 在 interaction boundary 套用外部事件。
- `src/tau2/domains/retail_stock/environment.py`
  - 從 task initial state 建立事件排程和隔離 DB。
- `data/tau2/domains/retail_stock/tasks.json`
  - 加入 task families 與參數化案例。
- `data/tau2/domains/retail_stock/db.json`
  - 加入多商品與完整 inventory record。
- `data/tau2/domains/retail_stock/policy.md`
  - 定義 read、reserve、commit 的能力邊界與失敗處理。

### 評估層

- 新增 `src/tau2/evaluator/evaluator_inventory.py`
  - DB invariant、event consistency、transaction correctness。
- 修改 `src/tau2/data_model/tasks.py`
  - 加入 inventory state assertions 和 transaction assertions。
- 修改 evaluator orchestration
  - 將 `INVENTORY_SAFETY` 納入 reward breakdown。

若只是擴充現有 `retail_stock`，Registry 不需要再次修改；只有新增另一個 domain 名稱時才需要註冊。

## 13. 測試計畫

### Unit tests

- Stock read 不會修改狀態。
- 正確 version 且庫存足夠時 reservation 成功。
- Stale version 時 reservation 失敗且 DB 完全不變。
- 庫存不足時 reservation 失敗且 DB 完全不變。
- 同一 reservation 不能被 commit 兩次。
- 取消／到期會正確釋放 reserved quantity。
- 每次 mutation 正確遞增 version。
- 事件只影響指定 product／item。

### Concurrency tests

- 兩個執行緒同時 reserve 最後庫存，最多一個成功。
- 高次數併發後所有 invariant 仍成立。
- 隨機 interleaving 可由 seed 重現。

### Evaluator tests

- DB oversell 即使回答文字漂亮也必須失敗。
- DB 安全但 agent 虛假宣稱成功時，`COMMUNICATE` 必須失敗。
- Transaction 成功且回答 grounded 時全部通過。
- 非英文但結構化事實正確的回答可通過。

### End-to-end tests

- 用 scripted agent 精確覆蓋每個 task family 的成功與失敗路徑。
- 再用實際 LLM 跑 smoke evaluation，但 CI 不依賴外部 API。

## 14. 分階段實作

### Phase 1：修正模擬語意

- 將背景事件移出 read tool。
- 建立 `ScheduledEvent` 與 event scheduler。
- 讓 `get_variant_stock` 完全 read-only。
- 保留現有兩個 task 作為回歸測試。

完成條件：讀取本身不改 DB；相同 seed 和 scenario 仍完全可重現。

### Phase 2：加入真實交易能力

- 實作 inventory、reservation、order models。
- 實作 atomic versioned reservation。
- 實作 create／cancel／expire workflow。
- 加入 transaction audit log。

完成條件：單元測試證明 stale 或 insufficient transaction 不會修改狀態。

### Phase 3：建立安全性 evaluator

- 新增 DB invariant evaluator。
- 新增 transaction-grounded communication checks。
- 將安全性設為 primary endpoint。

完成條件：刻意注入 oversell bug 時，測試必須可靠地失敗。

### Phase 4：擴充 task matrix

- 建立 A–H task families。
- 參數化數量、時機與競爭者。
- 建立 control／treatment variants。

完成條件：不能靠記住固定的 8、2、5、3 通過資料集。

### Phase 5：模型實驗與報告

- 選定模型與固定設定。
- 多 seed、多 trial 執行。
- 報告 oversell、false commitment、recovery 與 fulfillment 指標。
- 保存每次失敗的 trajectory、event log 和最終 DB state。

完成條件：主張與觀察指標一一對應，結果可由保存的 artifact 重現。

## 15. 支持強主張的最低驗收標準

在聲稱「系統能避免庫存 overcommitment」之前，至少必須滿足：

1. 在包含競爭交易的完整測試矩陣中，狀態層 oversell 為 0。
2. Stale version 的 reservation 成功數為 0。
3. 失敗 transaction 造成的部分 state mutation 為 0。
4. 重複 commit 造成的重複扣庫存為 0。
5. 每個成功 order 都可追溯至一個有效且唯一的 reservation。
6. False commitment rate 被單獨量測，而不是混入平均 reward。
7. 對照組能實際重現預期的 race／overpromise 失敗，證明 benchmark 有能力偵測問題。
8. 所有 trial 保存可重播的 scenario、seed、trajectory、event log 與 DB final state。

若只達成第 1–5 點，可以主張交易工具維持安全 invariant；若還要主張「agent-system 整體可靠」，則第 6–8 點以及足夠的模型實驗樣本也不可缺少。

## 16. 最終建議

不要把更多 regex 或固定數字堆在目前 evaluator 上作為主要改進。優先順序應是：

```text
純讀取 stock snapshot
→ 獨立外部事件
→ 原子且帶版本的 reservation
→ 真實 order commit
→ DB invariant evaluator
→ 多情境對照實驗
→ 語言 groundedness 評估
```

這樣才能把研究中心從「agent 有沒有說對答案」移到「即使 agent 使用陳舊資訊，底層交易也不能超賣；而 agent 必須如實描述交易結果」。

