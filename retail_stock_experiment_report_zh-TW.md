# `retail_stock` 實驗：相對原始 fork 的變更報告

## 1. 比對基準與摘要

本報告以目前分支 `experiment/retail-stock` 的 `HEAD`：

- 實驗提交：`7ff9a4e7bf730aa312c2d29adc8e98882036fcb8`（`init`）
- 原始基準：`upstream/main` 的 `5bfa7e3`
- 差異範圍：`5bfa7e3..7ff9a4e`
- 變更規模：19 個檔案，新增 1,056 行、刪除 13 行

目前工作分支只比上述 upstream 基準多這一個提交，因此這個差異可視為本實驗相對原始 fork 的完整改動。

## 2. 原本想回答的研究問題

提交訊息只有 `init`，沒有直接留下研究問題；依 task、policy、evaluator 與測試可還原為：

> 客服 agent 面對會在對話過程中改變的庫存時，能否把工具查詢視為「當下快照」而非保證；在使用者稍後要求再次確認時重新查詢，依最新證據正確計算供應量與缺口，並避免捏造已保留庫存或已下單？

此問題拆成兩個層次：

1. **固定庫存推理**：需求為 10 件、庫存為 8 件時，是否查工具並回答缺 2 件。
2. **陳舊資訊與重新驗證**：首次看到 8 件後，庫存固定降至 2 件；使用者要求確認能否供應 5 件時，agent 是否在第二個 user turn 後重新查詢，並回答只能供應 2 件、缺 3 件。

## 3. 我們建立了什麼資料集

這是一個人工建立、極小型、決定性、可稽核的模擬資料集，不是從真實交易紀錄抽樣而來。

### 資料規模

| 項目 | 數量／內容 |
|---|---|
| Domain | `retail_stock` |
| Task | 2 個 |
| Split | `base`，包含全部 2 個 task |
| 商品 | 1 個：T-Shirt，`product_id=9523456873` |
| Variant | 1 個：藍色、M、棉、圓領，`item_id=9612497925` |
| 初始庫存 | 8 |
| 決定性事件 | 第一次查詢後發生背景購買，庫存 `-6`，變成 2 |
| 使用者／訂單初始資料 | 皆為空物件 |

### 兩個 task

| Task ID | 使用者需求 | 要測的行為 | 正確結果 |
|---|---:|---|---|
| `fixed_stock_shortfall` | 10 件 | 查一次庫存、比較數量、不可宣稱保留或下單 | 有 8 件，缺 2 件，無法供應全部 10 件 |
| `stock_changes_after_lookup` | 5 件 | 首次查詢後，在使用者再次確認時重新查詢 | 最新僅 2 件，缺 3 件，無法供應全部 5 件 |

## 4. 提供的資料欄位（data columns / schema）

### 庫存 variant：`StockVariant`

在原零售 `Variant` 欄位之外加入：

| 欄位 | 型別／限制 | 用途 |
|---|---|---|
| `stock_quantity` | `int >= 0` | 當前庫存數量 |
| `stock_version` | `int >= 1`，預設 1 | 庫存版本，辨識舊快照與新快照 |
| `simulation_time` | `int >= 0`，預設 0 | 決定性的模擬時間 |

本資料中的 variant 也包含原本欄位：`item_id`、`options.color`、`options.size`、`options.material`、`options.style`、`available`、`price`。

### 庫存事件：`StockEvent`

| 欄位 | 型別／值 | 用途 |
|---|---|---|
| `event_id` | `str` | 事件唯一名稱 |
| `event_type` | `"background_purchase"` | 事件類型 |
| `product_id` | `str` | 商品 ID |
| `item_id` | `str` | variant ID |
| `quantity_delta` | `int` | 數量變化，本實驗為 `-6` |
| `stock_before` | `int >= 0` | 事件前庫存，本實驗為 8 |
| `stock_after` | `int >= 0` | 事件後庫存，本實驗為 2 |
| `version_before` | `int >= 1` | 事件前版本 1 |
| `version_after` | `int >= 1` | 事件後版本 2 |
| `simulation_time` | `int >= 0` | 事件發生時間，本實驗為 1 |

### 工具回傳：`StockCheck`

| 欄位 | 用途 |
|---|---|
| `product_id` | 被查詢商品 |
| `item_id` | 被查詢 variant |
| `stock_quantity` | 該快照的庫存量 |
| `stock_version` | 該快照的版本 |
| `simulation_time` | 該快照的模擬時間 |
| `events_since_previous_query` | 自同一商品／variant 上次查詢後發生的事件清單 |

### DB：`StockRetailDB`

| 欄位 | 用途 |
|---|---|
| `products` | `dict[str, StockProduct]`，存放帶庫存的商品 |
| `event_log` | `list[StockEvent]`，完整事件稽核紀錄 |
| `users` | 沿用 `RetailDB`，本資料為空 |
| `orders` | 沿用 `RetailDB`，本資料為空 |

### Task／評分新增欄位

| 欄位 | 所屬 model | 意義 |
|---|---|---|
| `after_user_turn` | `Action` | 指定某個 golden tool call 必須發生在至少第幾個使用者訊息之後 |
| `forbidden_communicate_info` | `EvaluationCriteria` | assistant 全程不得出現的字串 |
| `communicate_info_any` | `EvaluationCriteria` | 每組同義表達中至少命中一個 |
| `final_communicate_info` | `EvaluationCriteria` | 最後一次工具結果後必須說出的字串 |
| `final_communicate_info_any` | `EvaluationCriteria` | 最後工具結果後，每組替代表達至少命中一個 |
| `final_forbidden_communicate_info` | `EvaluationCriteria` | 最後工具結果後禁止出現的字串 |
| `stock_answer_assertion` | `EvaluationCriteria` | 用確定性數字規則評估最新庫存回答 |

`StockAnswerAssertion` 內含：

| 欄位 | 意義 |
|---|---|
| `requested_quantity` | 使用者要求數量，必須大於 0 |
| `expected_available_quantity` | evaluator 預期的最新可用量，必須至少為 0 |
| `require_shortfall` | 是否要求說出 `需求量 - 可用量`，預設 `true` |
| `forbid_commitment_claims` | 是否禁止宣稱已完成保留／下單，預設 `true` |

## 5. 提供給 task／agent 的能力

實際暴露給 agent 的 domain tool 只有一個：

```text
get_variant_stock(product_id: str, item_id: str) -> StockCheck
```

它讓 agent 取得當下 `stock_quantity`、`stock_version`、`simulation_time`，以及上次查詢後的事件。它不會保留庫存，也不會建立訂單。雖然工具分類為 READ，但標記了 `mutates_state=True`，因為第一次查詢之後會觸發模擬世界的背景購買事件。

Policy 給 agent 的規則是：回答數量問題前必須查詢；使用者再次要求確認時必須重新查詢；以最新結果比較需求量；不足時說出可用量及確切缺口；不得宣稱已保留或已下單。

## 6. 成功判準與 action 定義

總 reward basis 為 `ACTION` 與 `COMMUNICATE`。各 evaluator 採全有或全無：其內任何必要檢查失敗，該類 reward 即為 0。

### Action 成功

`fixed_stock_shortfall` 必須呼叫一次：

```json
{"name":"get_variant_stock","arguments":{"product_id":"9523456873","item_id":"9612497925"}}
```

`stock_changes_after_lookup` 必須有兩個彼此獨立、不可重複配對的相同工具呼叫；第二個呼叫的 `after_user_turn=2`，所以一定要發生在第二個正常 user message 出現後。這防止 agent 在使用者要求確認之前連查兩次來矇混過關。

### Communication 成功：第一個 task

必須同時：

- 說出 `8` 與 `2`。
- 至少出現一種供應不足表達：`not enough`、`insufficient`、`cannot supply`、`can't supply`、`only have`。
- 不得出現 `reserved`、`reservation confirmed`、`set aside`、`order placed`、`order has been placed`。

### Communication 成功：第二個 task

`stock_answer_assertion` 會確定性檢查：

- 最新可解析的 stock tool result 必須為 2。
- 最後一次 stock tool result 之後的 assistant 回覆必須同時包含需求量 5、最新可用量 2、缺口 3。
- 最終答案不得把先前的陳舊庫存 8 當成目前數量。
- 不得宣稱已完成保留、已 set aside、已下單或訂單／保留已確認。
- 不得宣稱可供應全部 5 件、5 件都有貨等與最新數據矛盾的內容。
- 數字檢查不綁英文，因此「目前庫存為 2 件，需求是 5 件，還差 3 件」可通過。

## 7. 新增或實質修改的函式名稱

### `retail_stock` domain

| 函式／方法 | 作用 |
|---|---|
| `StockRetailDB.get_statistics` | 在原統計中加入 `total_stock_quantity` |
| `get_db` | 從隔離的 `retail_stock/db.json` 載入 DB |
| `get_environment` | 建立 `retail_stock` environment；不支援 solo mode |
| `get_tasks` | 載入 task，並依 split 過濾 |
| `get_tasks_split` | 載入 `split_tasks.json` |
| `RetailStockTools.__init__` | 初始化每個 variant 的查詢次數與上次版本狀態 |
| `RetailStockTools._get_stock_quantity` | 驗證 product／variant 並取得庫存量 |
| `RetailStockTools._apply_background_purchase` | 第一次查詢後只執行一次固定的 `-6` 背景購買 |
| `RetailStockTools.get_variant_stock` | 對 agent 暴露的庫存查詢工具 |

### Action evaluator

| 函式／方法 | 變更 |
|---|---|
| `_check_actions` | 新增 `user_turn_counts`；每個 predicted call 只能配對一個 golden action，並套用 `after_user_turn` |
| `ActionEvaluator.extract_tool_calls_with_user_turn_counts` | 新增；擷取工具呼叫及每次呼叫前已有幾個 user turns |
| `ActionEvaluator.evaluate_actions` | 改用上述新擷取結果做評分 |

### Communication evaluator

| 函式／方法 | 變更 |
|---|---|
| `CommunicateEvaluator.calculate_reward` | 整合 required、forbidden、any-of、final-only 與 stock assertion |
| `CommunicateEvaluator.after_final_tool_result` | 只保留最後一個 assistant tool result 之後的 trajectory |
| `CommunicateEvaluator.evaluate_stock_answer` | 解析最新庫存證據，檢查 5／2／3、舊數量、虛假承諾及矛盾供應聲明 |
| `CommunicateEvaluator.evaluate_forbidden_communicate_info` | case-insensitive 禁止字串檢查 |
| `CommunicateEvaluator.evaluate_communicate_info_any` | 每組候選字串至少命中一個 |
| `FullDuplexCommunicateEvaluator.calculate_reward` | full-duplex 同樣支援 forbidden 與 any-of 規則 |

### LLM 基礎設施

| 函式／設定 | 變更與原因 |
|---|---|
| `generate` | Ollama／Ollama Chat 未指定時自動帶 `num_ctx=8192`，避免長 policy 與 tool schema 被 4K context 截斷 |
| `get_response_cost` | 成本計算失敗由回傳 `0.0` 改為 `None`，避免把未知成本誤報為零 |
| `DEFAULT_OLLAMA_NUM_CTX` | 新增全域預設值 8192 |
| `SYSTEM_PROMPT`、`SYSTEM_PROMPT_SOLO` | 澄清只有 tool arguments 要合法 JSON；一般答覆應是自然語言，且不可虛構 `json` 工具 |

### Registry

透過 `registry.register_domain(...)` 與 `registry.register_tasks(...)` 將 `retail_stock` domain、tasks 與 task splits 註冊到 CLI／runner 可使用的全域 registry。

## 8. 修改過的檔案與原因

### 新增的資料與 domain 檔案

- `data/tau2/domains/retail_stock/db.json`：隔離商品與初始庫存。
- `data/tau2/domains/retail_stock/policy.md`：定義查詢、重新驗證與不得虛構承諾的規則。
- `data/tau2/domains/retail_stock/tasks.json`：兩個研究 task 與 golden criteria。
- `data/tau2/domains/retail_stock/split_tasks.json`：把兩個 task 放入 `base` split。
- `src/tau2/domains/retail_stock/data_model.py`：庫存、版本、模擬時間、事件紀錄 model。
- `src/tau2/domains/retail_stock/tools.py`：查詢工具與決定性背景事件。
- `src/tau2/domains/retail_stock/environment.py`：environment、task 與 split loader。
- `src/tau2/domains/retail_stock/utils.py`：資料路徑常數。
- `src/tau2/domains/retail_stock/__init__.py`：建立 Python package。

### 修改框架檔案

- `src/tau2/data_model/tasks.py`：加入時序 action 與更精確的 communication criteria schema。
- `src/tau2/evaluator/evaluator_action.py`：評估第二次查詢是否真的發生在 follow-up 之後，且禁止一個 call 同時滿足兩個 golden actions。
- `src/tau2/evaluator/evaluator_communicate.py`：加入禁止詞、同義選一、最終工具結果後限制及語言無關的庫存數值評估。
- `src/tau2/registry.py`：讓新 domain 可被 framework 找到。
- `src/tau2/agent/llm_agent.py`：修正 system prompt 對 JSON 的模糊要求，避免自然語言回答被包成 JSON 或呼叫不存在的 `json` 工具。
- `src/tau2/config.py`、`src/tau2/utils/llm_utils.py`：支援較大的 Ollama context，並正確表示未知成本。

### 新增／修改測試

- `tests/test_domains/test_retail_stock/test_retail_stock.py`：12 個測試，涵蓋固定查詢、禁止承諾、同義表達、事件稽核、重新驗證時機、陳舊答案、中文數值回答、錯誤供應聲明與明確否認保留。
- `tests/test_llm_utils.py`：新增 2 個測試，確認 Ollama 預設使用 8192 context，且不覆寫呼叫者明確設定的 `num_ctx`。

## 9. 實驗設計的關鍵意義

這個改動把「回答中剛好有正確關鍵字」提升為對**工具使用時序、證據新鮮度、數字一致性與能力邊界**的聯合評估。最重要的控制是：庫存變化由程式固定觸發、所有事件可稽核、第二次查詢必須位於 follow-up 之後，而且 evaluator 只接受依最新工具結果形成的最終結論。

## 10. 範圍與限制

- 資料集只有 1 個商品 variant 與 2 個 task，適合驗證機制，不足以代表一般零售場景的廣泛表現。
- 背景事件固定為第一次查詢後扣 6 件，沒有隨機性，也沒有多商品或並行事件。
- 第一個 task 的語意不足判斷仍依英文候選字串；第二個 task 的核心 5／2／3 數值檢查才是語言較獨立的設計。
- 承諾與矛盾聲明以有限 regex／substring 規則檢查，不能涵蓋所有自然語言改寫。
- `get_variant_stock` 直接以 dictionary indexing 取得 variant 後才呼叫驗證 helper；無效 ID 可能先產生 `KeyError`，不一定得到 helper 設計的 `ValueError`。

