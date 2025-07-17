import os
import sys
import time
import csv
# import resource # Xóa hoặc comment dòng này
import psutil   # Thêm dòng này
from copy import deepcopy

# ... (Phần import TwoDKEDA và read_data giữ nguyên) ...
try:
    from TwoDKEDA import TwoDKnapsackEDAAlgorithm
    from read_data import read_data
except ImportError as e:
    print(f"Lỗi import: {e}. Đảm bảo TwoDKEDA.py và read_data.py ở đúng vị trí.")
    sys.exit(1)

def run_single_instance(data_file_path, eda_params_template):
    """
    Chạy thuật toán EDA cho một instance và trả về kết quả.
    """
    print(f"\n--- Đang xử lý instance: {data_file_path} ---")

    try:
        num_item_classes, H_strip, W_strip, heights, widths, unit_utilities = read_data(data_file_path)
    except Exception as e:
        print(f"Lỗi khi đọc file {data_file_path}: {e}")
        return None

    all_rectangles_data = []
    for i in range(num_item_classes):
        rect_profit = unit_utilities[i] * heights[i]
        all_rectangles_data.append({
            'id': i, 'type_id': i, 'width': widths[i],
            'height': heights[i], 'profit': rect_profit
        })

    print(f"  Đã đọc {len(all_rectangles_data)} vật phẩm. Thùng W={W_strip}, H={H_strip}")

    current_eda_params = deepcopy(eda_params_template)
    num_rects_master = len(all_rectangles_data)
    if num_rects_master < 100:
        current_eda_params['LSremn_iters'] = eda_params_template.get('LSremn_iters_small_n', 5)
        current_eda_params['generations_limit'] = eda_params_template.get('generations_limit_small_n', 20)
    else:
        current_eda_params['LSremn_iters'] = eda_params_template.get('LSremn_iters_large_n', 2)
        current_eda_params['generations_limit'] = eda_params_template.get('generations_limit_large_n', 15)
        if num_rects_master > 500:
             current_eda_params['generations_limit'] = eda_params_template.get('generations_limit_very_large_n', 10)

    # Đo thời gian
    start_time = time.perf_counter()

    # Lấy thông tin tiến trình hiện tại để theo dõi bộ nhớ
    current_process = psutil.Process(os.getpid())
    # Ghi lại bộ nhớ sử dụng ban đầu (Resident Set Size - RSS)
    # mem_before_rss = current_process.memory_info().rss
    # Việc theo dõi peak memory CHỈ cho phần solver mà không dùng profiler chuyên dụng
    # hoặc chạy solver trong process riêng là khó.
    # psutil.Process().memory_info().rss sẽ cho bạn mức sử dụng hiện tại.
    # Chúng ta sẽ lấy giá trị này sau khi solver chạy xong, nó sẽ gần với peak
    # nếu solver là phần tốn bộ nhớ nhất.
    # Một cách tốt hơn là theo dõi liên tục hoặc dùng memory_profiler.
    # Tuy nhiên, để đơn giản, chúng ta sẽ lấy memory info sau khi chạy.

    solver = TwoDKnapsackEDAAlgorithm(all_rectangles_data, W_strip, H_strip, current_eda_params)
    final_best_solution = solver.run_evolution_loop()

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # Lấy mức sử dụng bộ nhớ (RSS) sau khi chạy
    # memory_info() trả về bytes, chúng ta chuyển đổi sang MB
    peak_memory_mb = current_process.memory_info().rss / (1024 * 1024) 
    # Lưu ý: Đây là RSS tại thời điểm gọi, không hoàn toàn là "peak" trong suốt quá trình chạy solver,
    # nhưng là một ước lượng tốt nếu solver là phần chính tiêu thụ bộ nhớ.
    # Đối với peak memory thực sự, `resource.getrusage` tốt hơn trên Unix,
    # hoặc dùng `memory_profiler` cho chi tiết hơn.

    print(f"  Hoàn thành. Thời gian: {execution_time:.2f}s, Bộ nhớ (RSS): {peak_memory_mb:.2f}MB")

    results = {
        "instance_name": os.path.basename(data_file_path),
        "n_items": num_item_classes,
        "W_strip": W_strip,
        "H_strip": H_strip,
        "fitness": final_best_solution.fitness if final_best_solution else -1,
        "total_allocated_height": final_best_solution.total_allocated_height if final_best_solution else -1,
        "num_layers": len(final_best_solution.layers) if final_best_solution else 0,
        "num_rem_set": len(final_best_solution.rem_set) if final_best_solution else num_item_classes,
        "execution_time_s": execution_time,
        "peak_memory_mb": peak_memory_mb, # Đã đổi tên để phản ánh RSS
        "generations_run": current_eda_params['generations_limit']
    }
    return results

# ... (Phần if __name__ == "__main__": giữ nguyên như trước, không cần thay đổi) ...

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cách sử dụng: python experiment_runner.py <data_directory> <results_csv_file>")
        print("Ví dụ: python experiment_runner.py BAZAAR results/bazaar_results.csv")
        sys.exit(1)

    data_dir = sys.argv[1]
    results_file = sys.argv[2]

    base_eda_parameters = {
        'p_block_building_prob': 0.5,
        't_pop_initial': 10,
        'tmax_pop_max': 20,
        'kn_loop_control': 5, 
        'alpha_learning': 0.2,
        'p_samp_vs_mutate': 0.8,
        'tp_selection_ratio': 0.2,
        'p_bf_heuristic_prob': 0.8,
        'p_impLS_prob': 0.1,
        'imax_fit_list': 100,
        'LSremn_iters_small_n': 10,
        'LSremn_iters_large_n': 5,
        'generations_limit_small_n': 50,
        'generations_limit_large_n': 30,
        'generations_limit_very_large_n': 20,
        'gp_restart_gens': 30,
        'rp_restart_ratio': 0.7,
    }

    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    instance_files = []
    if os.path.isdir(data_dir):
        for f_name in sorted(os.listdir(data_dir)):
            if f_name.startswith("bazaar-") and f_name.endswith(".txt"):
                try:
                    parts = f_name.replace(".txt", "").split('-')
                    n_val = int(parts[1])
                    seed_val = int(parts[2])
                    instance_files.append({'path': os.path.join(data_dir, f_name), 'n': n_val, 'seed': seed_val})
                except (IndexError, ValueError):
                    print(f"Không thể phân tích tên file: {f_name}, bỏ qua.")
                    continue
        instance_files.sort(key=lambda x: (x['n'], x['seed']))
    else:
        print(f"Thư mục dữ liệu {data_dir} không tồn tại.")
        sys.exit(1)

    if not instance_files:
        print(f"Không tìm thấy file instance nào trong {data_dir}.")
        sys.exit(1)

    print(f"Sẽ xử lý {len(instance_files)} instances...")

    fieldnames = [
        "instance_name", "n_items", "W_strip", "H_strip", 
        "fitness", "total_allocated_height", "num_layers", "num_rem_set",
        "execution_time_s", "peak_memory_mb", "generations_run"
    ]

    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance_info in instance_files:
            file_path = instance_info['path']
            result_data = run_single_instance(file_path, base_eda_parameters)
            if result_data:
                writer.writerow(result_data)
            csvfile.flush()

    print(f"\nHoàn tất! Kết quả đã được lưu vào: {results_file}")