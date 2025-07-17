# Giả sử bạn lưu TwoDKEDA.py và read_data.py trong cùng thư mục
# hoặc read_data.py có thể được import.

from TwoDKEDA import TwoDKnapsackEDAAlgorithm, Rectangle # Import các lớp cần thiết từ file của bạn
from read_data import read_data # Import hàm đọc dữ liệu
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_solution(solution_individual, strip_W, strip_H):
    if not solution_individual or not solution_individual.layers:
        print("Không có giải pháp hoặc không có lớp nào để vẽ.")
        return

    fig, ax = plt.subplots(1, figsize=(strip_W / 10, strip_H / 10)) # Điều chỉnh figsize nếu cần

    # Vẽ thùng chứa
    ax.add_patch(
        patches.Rectangle(
            (0, 0), strip_W, strip_H,
            facecolor='whitesmoke', edgecolor='black', linewidth=2
        )
    )

    # Gán màu cho các lớp hoặc hình chữ nhật (ví dụ)
    # Bạn có thể tạo danh sách màu đa dạng hơn
    layer_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow', 'lightpink', 'lightseagreen']
    rect_edge_color = 'dimgray'

    current_y_offset = 0
    for i, layer in enumerate(solution_individual.layers):
        layer_color = layer_colors[i % len(layer_colors)]
        
        # Vẽ ranh giới lớp (tùy chọn)
        ax.add_patch(
            patches.Rectangle(
                (0, layer.y_offset_on_strip), # y_offset_on_strip đã được tính
                layer.strip_width,
                layer.allocated_strip_height,
                facecolor='none', edgecolor='gray', linestyle='--', linewidth=0.8
            )
        )

        # Vẽ các hình chữ nhật trong lớp
        for rect in layer.rectangles:
            if rect.is_placed and rect.x_global is not None and rect.y_global is not None:
                ax.add_patch(
                    patches.Rectangle(
                        (rect.x_global, rect.y_global), # Sử dụng tọa độ toàn cục
                        rect.width,
                        rect.height,
                        facecolor=layer_color, # Hoặc mỗi hình một màu khác
                        edgecolor=rect_edge_color,
                        linewidth=1
                    )
                )
                # Thêm ID của hình chữ nhật (tùy chọn)
                ax.text(rect.x_global + rect.width / 2,
                        rect.y_global + rect.height / 2,
                        f"{rect.id}", # Hoặc f"ID:{rect.id}\nP:{rect.profit}"
                        ha='center', va='center', fontsize=6, color='black')
        
        current_y_offset += layer.allocated_strip_height


    ax.set_xlim(0, strip_W)
    ax.set_ylim(0, strip_H)
    ax.set_aspect('equal', adjustable='box') # Quan trọng để tỷ lệ đúng
    plt.title(f"Giải pháp Xếp hình 2D - Fitness: {solution_individual.fitness:.2f}")
    plt.xlabel(f"Chiều rộng Thùng (W = {strip_W})")
    plt.ylabel(f"Chiều cao Thùng (H = {strip_H})")
    plt.gca().invert_yaxis() # Đảo trục y để (0,0) ở góc trên bên trái nếu muốn (thường (0,0) ở dưới)
                           # Bỏ dòng này nếu muốn (0,0) ở góc dưới trái
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()

def main_test_runner(data_file_path):
    # Đọc dữ liệu từ file sử dụng read_data.py
    # (n_classes, bin_H, bin_W, item_heights, item_widths, item_unit_utilities)
    num_item_classes, H_strip, W_strip, heights, widths, unit_utilities = read_data(data_file_path) #

    # Chuyển đổi dữ liệu sang định dạng all_rectangles_data
    all_rectangles_data = []
    for i in range(num_item_classes):
        # Tính tổng lợi nhuận cho mỗi vật phẩm dựa trên unit_utility và chiều cao của nó
        # Điều này khớp với itemClass_weight = u[i]*h[i] trong mk_bazaar.py
        rect_profit = unit_utilities[i] * heights[i]
        
        all_rectangles_data.append({
            'id': i, # ID duy nhất cho mỗi loại vật phẩm (vì demand là 1)
            'type_id': i, # Coi mỗi item là một type riêng
            'width': widths[i],
            'height': heights[i],
            'profit': rect_profit 
        })

    print(f"Đã đọc {len(all_rectangles_data)} vật phẩm từ file: {data_file_path}")
    print(f"Kích thước thùng (Strip): W={W_strip}, H={H_strip}")

    # Các tham số cho thuật toán EDA (lấy từ lần thiết lập trước của bạn, có thể cần điều chỉnh)
    eda_parameters = {
        'p_block_building_prob': 0.5,
        't_pop_initial': 5,
        'tmax_pop_max': 10, # Giảm để chạy nhanh ví dụ
        'kn_loop_control': 3, # Giảm để chạy nhanh ví dụ
        'generations_limit': 15, # Giới hạn thế hệ cho ví dụ
        'alpha_learning': 0.2,
        'p_samp_vs_mutate': 1.0, # Có thể thay đổi giữa các lần chạy
        'tp_selection_ratio': 0.1,
        'p_bf_heuristic_prob': 0.8,
        'p_impLS_prob': 0.0, # Dựa trên một số thử nghiệm trước đó
        'imax_fit_list': 50,
        'LSremn_iters': 2, # Giảm cho ví dụ
        'gp_restart_gens': 20, # Giảm cho ví dụ
        'rp_restart_ratio': 0.7,
    }
    num_rects_master = len(all_rectangles_data)
    if num_rects_master < 100: eda_parameters['LSremn_iters'] = 5 
    else: eda_parameters['LSremn_iters'] = 2


    print("\nKhởi tạo Thuật toán 2DKEDA...")
    solver = TwoDKnapsackEDAAlgorithm(all_rectangles_data, W_strip, H_strip, eda_parameters)

    print("\nBắt đầu Vòng lặp Tiến hóa...")
    final_best_solution = solver.run_evolution_loop()

    print("\n--- Kết thúc Tiến hóa ---")
    if final_best_solution and final_best_solution.fitness > 0:
        print(f"Fitness của Giải pháp Tốt nhất: {final_best_solution.fitness:.2f}")
        # ... (các print khác của bạn) ...
        print(f"Số lượng Hình chữ nhật trong Rem Set: {len(final_best_solution.rem_set)}")
        
        # Gọi hàm vẽ
        plot_solution(final_best_solution, W_strip, H_strip) # W_strip, H_strip từ read_data
    else:
        print("Không tìm thấy giải pháp hiệu quả hoặc quần thể không được khởi tạo đúng cách.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách sử dụng: python your_main_script_name.py <path_to_bazaar_data_file.txt>")
        # Ví dụ: python your_main_script_name.py BAZAAR/bazaar-20-1.txt
        sys.exit(1)
    
    data_file = sys.argv[1]
    main_test_runner(data_file)