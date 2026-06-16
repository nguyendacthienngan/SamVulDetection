# Script phòng thủ — Kiến trúc & phương pháp đề xuất (CMV)
**Mục đích:** chuẩn bị cho câu hỏi xoáy sâu của phản biện về *tính khoa học của fusion* và *kiến trúc*. Số liệu đã verify với `Chapter3.tex` / `Chapter4.tex` / context canonical.

> Quy ước: **in đậm** = ý phải nhấn; công thức đọc chậm; khi bị hỏi xoáy — *theory-first* (nói WHY trước), rồi mới đưa bằng chứng thực nghiệm.

---

## A. BÀI NÓI LÕI (≈90s) — đọc khi trình bày slide kiến trúc/fusion

Kiến trúc CMV xử lý mã nguồn ở mức hàm qua **hai nhánh bổ sung nhau**. Nhánh đồ thị dùng Joern trích CPG rồi mã hóa bằng **GGNN** (ẩn 256, 3 lớp) — thu biểu diễn nút **H_G ∈ ℝ^{N×256}** mã hóa tường minh quan hệ điều khiển (CFG) và dữ liệu (DFG). Nhánh token dùng **CodeBERT đóng băng** — thu **H_T ∈ ℝ^{L×768}** giàu ngữ nghĩa từ vựng mà CPG sau chuẩn hóa dễ làm mất.

Đóng góp cốt lõi nằm ở **cách hợp nhất**. Trước hết chiếu cả hai về không gian chung **d_fuse = 512**. Sau đó **chú ý chéo hai chiều**: chiều token→graph cho mỗi token "đọc" toàn bộ nút đồ thị — `A_{T→G} = softmax(H_T'·H_G'ᵀ/√d)`, rồi `H̃_T = A_{T→G}·H_G'`; và chiều graph→token đối xứng — `A_{G→T} = softmax(H_G'·H_T'ᵀ/√d)`, `H̃_G = A_{G→T}·H_T'`. **Hai phương thức chủ động làm giàu lẫn nhau, thay vì chỉ ghép cạnh nhau như concatenation.** Cuối cùng mean-pool mỗi nhánh và hợp nhất có trọng số `h_fused = α·h_G^pool + β·h_T^pool`, với α+β=1.

Song song, một **Node Head** chấm điểm quan trọng s_v ∈ [0,1] cho từng nút, huấn luyện đồng thời với bộ phân loại qua hàm mất mát kết hợp `L = L_cls + λ·L_node` — chính việc huấn luyện đồng thời này khiến điểm quan trọng **bám tín hiệu phân loại (intrinsic)**, là nền tảng cho tính trung thực của giải thích.

---

## B. NGÂN HÀNG Q&A XOÁY SÂU (câu trả lời thủ sẵn)

### NHÓM 1 — Cơ chế fusion (trọng tâm phản biện)

**Q1. Cross-attention hai chiều của em khác gì co-attention trong ViLBERT/LXMERT? Đóng góp mới ở đâu?**
Em xin thẳng thắn: **bản thân cơ chế co-attention hai chiều không phải em phát minh** — em kế thừa từ dòng vision-language (LXMERT, ViLBERT). Đóng góp của luận văn là ở **ba chỗ khác**: (i) **đưa cơ chế này sang một cặp phương thức mới và dị thể** — đồ thị chương trình (GGNN, cấu trúc) với token mã nguồn (Transformer, từ vựng) — nơi hai bộ mã hóa khác bản chất; (ii) **gắn với trọng số bất đối xứng α>β có cơ sở lý thuyết** cho đặc thù lỗ hổng; (iii) **tích hợp Node Head huấn luyện đồng thời** để biến bộ phát hiện thành bộ phát hiện *có khả năng giải thích trung thực*. Em không nhận vơ phần cơ chế, mà nhận phần thích nghi và ứng dụng.

**Q2. Tại sao không dùng Graph Transformer / Graphormer xử lý chung graph và token bằng một self-attention?**
Hai lý do. **Thứ nhất, về độ phức tạp:** self-attention trên chuỗi nối tốn `O((N+L)²·d)`, bao gồm cả phần nội-đồ-thị `O(N²d)` và nội-token `O(L²d)` — **vốn đã được GGNN và CodeBERT xử lý rồi**, nên đó là tính toán dư thừa. Cross-attention của em chỉ tốn `O(N·L·d)`, **rẻ hơn và không lặp lại việc đã làm**. **Thứ hai, về inductive bias:** Graphormer mã hóa cấu trúc bằng positional/structural encoding — sẽ **trùng vai** với cơ chế truyền tin có cổng của GGNN. Em chọn để mỗi bộ mã hóa làm đúng việc nó mạnh, rồi chỉ cho chúng *tương tác* qua cross-attention.

**Q3. Em chứng minh hai chiều tốt hơn một chiều như thế nào?**
*Về lý thuyết:* tương tác một chiều chỉ làm giàu **một** phương thức, phương thức nguồn vẫn ở trạng thái nguyên bản trước khi hợp nhất; hai chiều đảm bảo **cả hai cùng được làm giàu đối xứng**. *Về thực nghiệm — đây là bằng chứng mạnh nhất:* trong ablation, **bỏ tính hai chiều (chuyển về một chiều) làm F1 giảm 0,360 — mức giảm LỚN NHẤT**, thậm chí lớn hơn cả việc **bỏ hẳn cross-attention (−0,304)**. Kết quả phản trực giác này nói lên: **chính tính đối xứng của trao đổi thông tin, chứ không phải độ phức tạp, mới là nguồn giá trị.**

**Q4. α=0,7 / β=0,3 có phải con số tùy tiện (heuristic) không? Sao không học adaptive?**
**Không tùy tiện.** Lý thuyết chỉ xác định **hướng α>β**, dựa trên ba lập luận hội tụ: (i) CPG mã hóa **tường minh** quan hệ DFG/CFG nên có **entropy điều kiện thấp hơn** chuỗi token vốn phải suy luận qua khoảng cách tuyến tính; (ii) Devign, ReVeal nhất quán cho thấy đặc trưng cấu trúc phân biệt lỗ hổng tốt hơn từ vựng thuần; (iii) CodeBERT đã hấp thụ một phần cấu trúc ngầm, nên GGNN cho **giá trị biên** cao hơn. Còn **giá trị cụ thể** thì em kiểm chứng bằng phân tích độ nhạy: quét α trên toàn dải, **F1 chỉ dao động 0,006 (σ=0,0021)** — tức có **vùng ổn định [0,55–0,85]**. Vậy 0,7 là **một lựa chọn có cơ sở nằm trong vùng robust**, không phải con số "ăn may". Em cố định thay vì học thích nghi vì: gating thích nghi trong thử nghiệm sơ bộ **không cải thiện đáng kể**, trong khi trọng số cố định **dễ diễn giải và ít tham số hơn** — học thích nghi là hướng tương lai.

**Q5. Vì sao mean-pooling sau cross-attention, không dùng attention-pooling hay token [CLS]?**
Vì **tương tác phân biệt đã xảy ra ở bước cross-attention rồi**; bước gộp chỉ cần tổng hợp ổn định. Mean-pooling **không thêm tham số, bền với N và L thay đổi**, tránh quá khớp trên dữ liệu mất cân bằng. Attention-pooling là một tinh chỉnh khả dĩ, nhưng em ưu tiên cấu hình tối giản, kiểm soát được.

### NHÓM 2 — Nhánh đồ thị & đặc trưng

**Q6. Tại sao GGNN chứ không GCN/GAT/GIN?**
Vì **cơ chế cập nhật có cổng kiểu GRU** giữ và lan truyền thông tin qua **T bước truyền tin**, nắm bắt **phụ thuộc cấu trúc tầm xa** trên CPG — trong khi GCN/GAT thường **over-smoothing** sau vài lớp. Mà nhiều lỗ hổng bản chất là **phụ thuộc giữa các câu lệnh ở xa nhau** trên luồng điều khiển/dữ liệu, nên cổng nhớ của GGNN là phù hợp nhất.

**Q7. Đặc trưng nút chỉ 2 chiều — không quá nghèo sao?**
Đây là **lựa chọn cố ý để tách bạch vai trò**: GGNN học từ **cấu trúc/topology**, còn ngữ nghĩa từ vựng do **CodeBERT** đảm nhận. Nếu nhồi đặc trưng nút giàu, ranh giới đóng góp giữa hai nhánh sẽ mờ, khó diễn giải ablation. Em có thừa nhận trong phần hạn chế rằng điều này có thể bỏ lỡ tín hiệu fine-grained — nhưng phân tích enrichment cho thấy **tín hiệu cấu trúc đang chiếm ưu thế** (cạnh REACHING_DEF làm giàu ~5× trên lớp lỗ hổng).

**Q8. GGNN 3 lớp có bị over-smoothing / thiếu năng lực trên đồ thị lớn không?**
Phân tích thất bại **bác bỏ giả thuyết này**: các đồ thị bị **bỏ sót (FN) có số nút trung bình NHỎ hơn (264)** so với đồ thị **dự đoán đúng (TP, 343 nút)**. Nếu là vấn đề quy mô/độ sâu thì FN phải là đồ thị lớn. Thực tế thất bại đến từ **tín hiệu liên thủ tục nằm NGOÀI phạm vi CPG mức hàm**, không phải năng lực GGNN.

### NHÓM 3 — CodeBERT đóng băng

**Q9. Đóng băng CodeBERT có giới hạn hiệu năng không? top-2 fine-tune +0,304 chẳng phải bằng chứng frozen kém?**
Nó cho thấy **chất lượng biểu diễn token là nút thắt — một dư địa cải thiện**, và em trình bày trung thực điều đó. Nhưng em **giữ frozen làm cấu hình chính thức** vì: kết quả top-2 (val F1 0,922 ở epoch 16) **chưa hội tụ, chưa kiểm chứng độc lập**, và có rủi ro **quá khớp** trên dữ liệu mất cân bằng/tăng cường RAG. Đóng băng cho **hiệu quả tham số** (125M đóng băng, chỉ ~1,2M huấn luyện — vừa một GPU phổ thông), **tránh quên thảm họa**, và **ổn định để tái lập**. Em xem fine-tune là *improvement gap* được báo cáo minh bạch, không phải kết quả chính.

### NHÓM 4 — Node Head & giải thích

**Q10. (Câu sắc nhất) Làm sao huấn luyện Node Head khi KHÔNG có nhãn mức nút?**
Bằng **giám sát yếu (weak supervision) từ nhãn mức đồ thị**. Hàm mất mát node là
`L_node = −(1/B) Σ_i (y_i/|V_i|) Σ_{v∈V_i} log(s_v + ε)`,
và **chỉ đóng góp khi y_i = 1** (mẫu có lỗ hổng) — nó **đẩy điểm s_v của các nút trong hàm lỗi lên cao**, huấn luyện đồng thời với mất mát phân loại. Em **không cần nhãn từng nút**. Tính trung thực đến từ chỗ điểm quan trọng được học **chung vòng** với tín hiệu phân loại (intrinsic), và được kiểm chứng *hậu kỳ* bằng ERASER Comprehensiveness 0,076 (cao hơn chọn ngẫu nhiên) và self-verify AGREE 75,8%.

**Q11. L_node không ràng buộc mẫu benign (y=0) — Node Head có gán điểm cao bừa cho hàm an toàn không?**
Đúng là **L_node chỉ giám sát mẫu dương**. Với mẫu an toàn, ràng buộc đến từ **mất mát phân loại**, từ bước **lọc dòng vô nghĩa** khi trích highlight, và từ **tầng tự xác minh bằng LLM độc lập**. Em cũng đã ghi rõ trong hạn chế rằng **chưa có negative control** (highlight ngẫu nhiên/đối kháng) để xác nhận — đó là hướng kiểm chứng tiếp theo. Em không giấu điểm này.

**Q11b. AGREE 75,8% — làm sao biết LLM thực sự *kiểm chứng* chứ không chỉ *gật đầu hợp lý hóa* (rationalize) mọi thứ?**
Đây là câu hỏi phương pháp luận đúng chỗ. Để khẳng định 75,8% có giá trị, cần một **đối chứng âm (negative control)**: đưa cho LLM các highlight **biết chắc là sai** rồi đo lại tỉ lệ AGREE. Hai loại đối chứng:
- **Random highlights:** chọn ngẫu nhiên các nút/dòng *không* phải do GGNN chấm cao.
- **Adversarial highlights:** cố tình chọn dòng *chắc chắn không liên quan* lỗ hổng (khai báo tầm thường, dấu ngoặc).

Kỳ vọng nếu hệ thống lành mạnh:

| Đầu vào | AGREE kỳ vọng |
|---|---|
| Highlight thật của GGNN | **Cao** (75,8% — đã đo) |
| Random highlights (đối chứng âm) | **Thấp** |
| Adversarial highlights (đối chứng âm) | **Rất thấp** |

Nếu AGREE của nhóm đối chứng âm cũng cao ngang nhóm thật → LLM chỉ rationalize, 75,8% **mất sức nặng**. Nếu thấp rõ rệt → **chứng minh LLM thực sự phân biệt** highlight tốt/xấu.

*Câu trả lời thủ sẵn:* "Em ý thức rằng AGREE 75,8% chỉ là bằng chứng đầy đủ nếu kèm **negative control** — tức kiểm chứng LLM cho điểm thấp với highlight ngẫu nhiên/đối kháng. Thí nghiệm này em **chưa kịp chạy** và đã nêu là hướng kiểm chứng tiếp theo trong phần hạn chế. Tuy nhiên em có **bằng chứng bổ trợ độc lập ở phía bộ phát hiện**: ERASER Comprehensiveness 0,076 > 0 cho thấy các dòng highlight thực sự *cần thiết* cho dự đoán hơn so với nút ngẫu nhiên — đó đã là một dạng đối chứng định lượng." → Biến điểm yếu thành thể hiện hiểu sâu phương pháp luận, thay vì lấp liếm.

**Q12. Vì sao tách detector và explainer, không làm end-to-end?**
Vì **tách biệt, không chia sẻ tham số** mới cho phép **đánh giá trung thực một cách khách quan**: mức đồng thuận giữa hai mô hình độc lập là bằng chứng thật, không phải mô hình "tự chấm điểm cho mình". Ngoài ra còn lợi về **mô-đun hóa** (thay LLM dễ dàng) và **tránh LLM bịa vị trí lỗi**. Đánh đổi là pipeline dài hơn — em chấp nhận để đổi lấy tính kiểm chứng được.

### NHÓM 5 — Độ phức tạp & dữ liệu (hỏi phụ)

**Q13. Chi phí cross-attention khi N, L lớn?**
Cross-attention là `O(N·L·d)`. Với CPG **mức hàm** (N bị chặn) và CodeBERT **cắt L ≤ 512**, chi phí kiểm soát được: suy luận **~0,14 ms/mẫu, >7.000 mẫu/giây, <600 MB** — rẻ hơn nhiều so với self-attention `O((N+L)²d)` của Graph Transformer. (Có slide chi phí suy luận dự phòng.)

**Q14. Cross-attention có chỉ "ăn may" trên MegaVul+ (dữ liệu synthetic) không?**
Không. Trên **BigVul — dữ liệu thực, không synthetic — F1 = 0,4756, vẫn vượt baseline MAPR +51,8% tương đối**. Đóng góp fusion **vững trên cả dữ liệu thực**, nên không phải hiệu ứng của tăng cường tổng hợp.

---

## C. CHEAT-SHEET một câu (khi cần phản xạ nhanh)
- **Fusion khác concat:** concat chỉ *ghép*; cross-attention cho hai phương thức *hỏi* nhau (`softmax(QKᵀ/√d)`).
- **Hai chiều > một chiều:** ablation ΔF1 = **−0,360** (lớn nhất) → đối xứng là nguồn giá trị.
- **α>β có cơ sở:** entropy điều kiện thấp + Devign/ReVeal + CodeBERT đã có cấu trúc ngầm; robust [0,55–0,85], spread 0,006.
- **Node Head không cần nhãn nút:** weak supervision, L_node chỉ kích hoạt khi y=1.
- **Frozen CodeBERT:** chính thức vì ổn định/tái lập; fine-tune +0,304 là *gap*, chưa hội tụ.
- **GGNN:** cổng GRU giữ phụ thuộc tầm xa, chống over-smoothing.
- **Thất bại không do quy mô:** FN (264 nút) < TP (343 nút) → lỗi ở phạm vi liên thủ tục, không ở GGNN.
