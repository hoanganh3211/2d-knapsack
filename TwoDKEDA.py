import random
import numpy as np
from copy import deepcopy
import itertools
# --- Cấu trúc dữ liệu cơ bản ---
class Rectangle:
    def __init__(self, id, width, height, profit=None, type_id=None, is_block_type=0):
        self.id = id # ID duy nhất cho mỗi instance hình chữ nhật
        self.type_id = type_id if type_id is not None else id # ID loại gốc
        self.width = width
        self.height = height
        # Lợi nhuận Pi bằng diện tích (không trọng số) hoặc độc lập (có trọng số) [cite: 17, 18]
        self.profit = profit if profit is not None else width * height
        self.x = None # Tọa độ x tương đối trong lớp (layer)
        self.y = None # Tọa độ y tương đối trong lớp (layer)
        self.x_global = None # Tọa độ x toàn cục trên dải (strip)
        self.y_global = None # Tọa độ y toàn cục trên dải (strip)
        self.is_placed = False # Đã được đặt trong một lớp hay chưa
        self.is_block_type = is_block_type # 0: gốc, 1: 2x1, 2: 1x2, 3: 2x2 (ví dụ) [cite: 232]

    def __repr__(self):
        return (f"Rect(id={self.id}, w={self.width}, h={self.height}, p={self.profit}, "
                f"layer_pos=({self.x},{self.y}), placed={self.is_placed})")

class Layer:
    def __init__(self, layer_id, strip_width, allocated_strip_height):
        self.id = layer_id
        self.strip_width = strip_width # Chiều rộng của dải gốc W
        self.allocated_strip_height = allocated_strip_height # Chiều cao được cấp phát cho lớp này
        self.content_actual_height = 0 # Chiều cao thực tế mà các hình chữ nhật chiếm dụng
        self.rectangles = [] # Danh sách các Rectangle objects đã được đặt trong lớp
        self.cutting_commands = [] # Danh sách các lệnh cắt cho lớp này
        self.total_profit = 0
        self.y_offset_on_strip = 0 # Tọa độ y bắt đầu của lớp này trên dải chính

    def calculate_content_dims_and_profit(self):
        self.content_actual_height = 0
        self.total_profit = 0
        if not self.rectangles:
            return 0, 0
        max_y_coord_in_layer = 0
        for r in self.rectangles:
            if r.is_placed and r.y is not None and r.height is not None:
                if r.y + r.height > max_y_coord_in_layer:
                    max_y_coord_in_layer = r.y + r.height
                self.total_profit += r.profit
        self.content_actual_height = max_y_coord_in_layer
        return self.content_actual_height, self.total_profit

class SolutionIndividual: # Đại diện cho một cá thể trong quần thể [cite: 205]
    def __init__(self, strip_width, strip_height_max):
        self.strip_width = strip_width
        self.strip_height_max = strip_height_max # Chiều cao tối đa của dải chính H
        self.layers = [] # Danh sách các Layer objects [cite: 207]
        self.rem_set = [] # Các hình chữ nhật không nằm trong knapsack [cite: 206]
        self.fitness = 0 # Tổng lợi nhuận của các hình chữ nhật trong các lớp [cite: 212]
        self.total_allocated_height = 0 # Tổng chiều cao được cấp phát của các lớp

    def calculate_fitness_and_total_height(self): # Tính toán lại fitness và tổng chiều cao
        self.fitness = 0
        self.total_allocated_height = 0
        current_y_offset = 0
        for layer in self.layers:
            layer.y_offset_on_strip = current_y_offset # Gán vị trí bắt đầu y của lớp trên dải
            _, profit = layer.calculate_content_dims_and_profit()
            self.fitness += profit
            self.total_allocated_height += layer.allocated_strip_height
            current_y_offset += layer.allocated_strip_height
            # Cập nhật tọa độ toàn cục cho các hình chữ nhật trong lớp
            for r in layer.rectangles:
                if r.is_placed:
                    r.x_global = r.x # Giả sử x trong lớp là toàn cục hoặc bắt đầu từ 0
                    r.y_global = layer.y_offset_on_strip + r.y
        return self.fitness

    def is_feasible(self): # Kiểm tra xem tổng chiều cao có vượt quá H không
        return self.total_allocated_height <= self.strip_height_max

    def get_all_placed_rect_ids(self): # Lấy ID của tất cả các hình chữ nhật đã được đặt
        ids = set()
        for layer in self.layers:
            for r in layer.rectangles:
                if r.is_placed:
                    ids.add(r.id)
        return ids

# --- Thủ tục PackLayer (Algorithm 1 trong bài báo) ---
class PackLayerProcedure:
    def __init__(self, params):
        self.p_bf = params.get('p_bf_heuristic_prob', 0.8) # Xác suất sử dụng HP1 (best-fit) [cite: 117]
        self.p_imp = params.get('p_impLS_prob', 0.0) # Xác suất sử dụng ImpLS [cite: 118]
        self.imax_fit_list_length = params.get('imax_fit_list', 200) # Độ dài tối đa của fit_list [cite: 107]

        # Các biến trạng thái cho một lần chạy xếp lớp
        self._Q_map = {} # Map: rect_id -> Rectangle object cho lớp hiện tại
        self._placed_rects_current_layer = [] # Danh sách các hình chữ nhật đã đặt trong lớp hiện tại
        self._cutting_commands_current_layer = [] # Lệnh cắt cho lớp hiện tại
        self._min_dw_threshold_current_Q = 1 # Ngưỡng cho dw, tính dựa trên Q hiện tại

    def _get_fit_list(self, region_w, region_h): #
        fit_list = []
        for r_obj in self._Q_map.values():
            if not r_obj.is_placed and r_obj.width <= region_w and r_obj.height <= region_h:
                fit_list.append(r_obj)
                if len(fit_list) >= self.imax_fit_list_length:
                    break
        return fit_list

    def _get_best_fit_rectangle_from_Q(self, region_w, region_h): #
        best_r_obj = None
        max_profit_found = -1.0
        # Quan trọng: Bài báo ghi "If there are more best-fit rectangles it selects one of them with the largest width."
        # Điều này ngụ ý rằng chúng ta tìm profit lớn nhất trước, sau đó mới xét width.

        # Tìm profit lớn nhất trong số các hình chữ nhật phù hợp
        for r_obj in self._Q_map.values():
            if not r_obj.is_placed and \
            r_obj.width <= region_w and \
            r_obj.height <= region_h:
                if r_obj.profit > max_profit_found:
                    max_profit_found = r_obj.profit

        # Nếu không tìm thấy hình chữ nhật nào có profit > -1 (tức là có ít nhất một hình phù hợp)
        if max_profit_found == -1.0:
            return None

        # Bây giờ tìm hình chữ nhật có profit bằng max_profit_found và có width lớn nhất
        max_width_at_max_profit = -1.0
        for r_obj in self._Q_map.values():
            if not r_obj.is_placed and \
            r_obj.width <= region_w and \
            r_obj.height <= region_h and \
            r_obj.profit == max_profit_found: # Chỉ xét những hình có profit lớn nhất
                if r_obj.width > max_width_at_max_profit:
                    max_width_at_max_profit = r_obj.width
                    best_r_obj = r_obj
        return best_r_obj

    def _apply_ImpLS_fit_block_search(self, 
                                    fit_list_remaining: list, # Danh sách các Rectangle object phù hợp, chưa được đặt
                                    available_width: float, 
                                    rect_height_for_block_alignment: float): #
        """
        Xây dựng các khối (blocks) kết hợp một, hai hoặc ba hình chữ nhật từ fit_list_remaining.
        Các hình chữ nhật được đặt kề nhau theo chiều rộng.
        Chiều cao của khối là chiều cao tối đa của các hình chữ nhật trong khối.
        Chọn khối phù hợp (vừa với available_width và rect_height_for_block_alignment)
        có tổng lợi nhuận lớn nhất.
        Trả về một list các Rectangle objects (tạo thành khối) hoặc None.
        """
        best_block_combination = None
        max_profit_of_best_block = -1.0

        # Đảm bảo fit_list_remaining không rỗng
        if not fit_list_remaining:
            return None

        # Thử khối 1 hình chữ nhật
        # "combining one..."
        for r1 in fit_list_remaining:
            if r1.width <= available_width and r1.height <= rect_height_for_block_alignment:
                if r1.profit > max_profit_of_best_block:
                    max_profit_of_best_block = r1.profit
                    best_block_combination = [deepcopy(r1)] # Trả về bản sao sâu

        # Thử khối 2 hình chữ nhật
        # "...two..."
        if len(fit_list_remaining) >= 2:
            for r1, r2 in itertools.combinations(fit_list_remaining, 2):
                current_block_width = r1.width + r2.width
                current_block_height = max(r1.height, r2.height) # "...maximum height of the rectangles in the block"

                if current_block_width <= available_width and \
                current_block_height <= rect_height_for_block_alignment:
                    current_block_profit = r1.profit + r2.profit
                    if current_block_profit > max_profit_of_best_block:
                        max_profit_of_best_block = current_block_profit
                        best_block_combination = [deepcopy(r1), deepcopy(r2)]

        # Thử khối 3 hình chữ nhật
        # "...or three rectangles"
        if len(fit_list_remaining) >= 3:
            for r1, r2, r3 in itertools.combinations(fit_list_remaining, 3):
                current_block_width = r1.width + r2.width + r3.width
                current_block_height = max(r1.height, r2.height, r3.height)

                if current_block_width <= available_width and \
                current_block_height <= rect_height_for_block_alignment:
                    current_block_profit = r1.profit + r2.profit + r3.profit
                    if current_block_profit > max_profit_of_best_block:
                        max_profit_of_best_block = current_block_profit
                        best_block_combination = [deepcopy(r1), deepcopy(r2), deepcopy(r3)]
        
        # if best_block_combination:
        #     print(f"    ImpLS: Found best block with profit {max_profit_of_best_block:.2f} "
        #           f"consisting of {len(best_block_combination)} rects.")
        # else:
        #     print(f"    ImpLS: No suitable block found.")
            
        return best_block_combination

    def _placement_strategy_and_ImpLS(self, fit_list, region_w, region_h): #
        # Trả về (item_or_block_to_place, combined_width, combined_height)
        # item_or_block_to_place có thể là một Rectangle hoặc list các Rectangles (block)
        
        selected_rect_for_placement = None
        # Placement Heuristics HP1 or HP2
        if random.random() < self.p_bf: # HP1: best-fit [cite: 117]
            selected_rect_for_placement = self._get_best_fit_rectangle_from_Q(region_w, region_h)
        elif fit_list: # HP2: random-fit from fit_list [cite: 117]
            selected_rect_for_placement = random.choice(fit_list)

        if not selected_rect_for_placement:
            return None, 0, 0

        # Giả sử ban đầu chỉ có một hình chữ nhật được chọn
        items_to_place = [selected_rect_for_placement]
        total_block_width = selected_rect_for_placement.width
        total_block_height = selected_rect_for_placement.height

        # Apply ImpLS local search [cite: 118]
        if random.random() < self.p_imp:
            remaining_width_for_block = region_w - total_block_width
            if remaining_width_for_block > 0:
                fit_list_for_impls = [r for r in fit_list if r.id != selected_rect_for_placement.id and not r.is_placed]
                
                # Chiều cao của khối ImpLS nên phù hợp với chiều cao của selected_rect_for_placement
                # hoặc ImpLS tự xác định chiều cao khối và sau đó total_block_height là max
                block_from_impls = self._apply_ImpLS_fit_block_search(
                    fit_list_for_impls,
                    remaining_width_for_block,
                    selected_rect_for_placement.height # Hoặc region_h nếu khối có thể cao hơn
                )
                if block_from_impls: # Nếu ImpLS tìm được khối
                    items_to_place.extend(block_from_impls)
                    total_block_width += sum(r.width for r in block_from_impls)
                    total_block_height = max(total_block_height, max(r.height for r in block_from_impls))
        
        return items_to_place, total_block_width, total_block_height

    def _execute_recursive_pack(self, vbw, vbh, x00, y00): # Triển khai Algorithm 1
        fit_list = self._get_fit_list(vbw, vbh) # [cite: 127] (implicit: check fit_list empty)
        if not fit_list: # "If the fit_list vector is empty then return fi" [cite: 127]
            return

        # "Apply our placement_strategy." [cite: 127] (bao gồm ImpLS)
        packed_items_list, plw, plh = self._placement_strategy_and_ImpLS(fit_list, vbw, vbh) # [cite: 128]

        if not packed_items_list: # Không chọn được item/block nào
            return

        # Đặt item/block (list các hình chữ nhật)
        current_x_offset_for_block = x00
        for item_rect in packed_items_list:
            item_rect.x = current_x_offset_for_block
            item_rect.y = y00
            item_rect.is_placed = True # Đánh dấu là đã đặt trong lớp này
            self._placed_rects_current_layer.append(item_rect)
            current_x_offset_for_block += item_rect.width
            # self._cutting_commands_current_layer.append(f"Place R{item_rect.id} at ({item_rect.x},{item_rect.y})")

        dw = vbw - plw # [cite: 129]

        # Tính ngưỡng dw tối thiểu dựa trên các hình chữ nhật còn lại trong Q
        self._min_dw_threshold_current_Q = float('inf')
        has_remaining_to_place = False
        for r_obj in self._Q_map.values():
            if not r_obj.is_placed:
                has_remaining_to_place = True
                self._min_dw_threshold_current_Q = min(self._min_dw_threshold_current_Q, r_obj.width)
        if not has_remaining_to_place: self._min_dw_threshold_current_Q = vbw + 1


        if dw < self._min_dw_threshold_current_Q: # "If (dw is too small width for other rectangle) then" [cite: 129]
            # "placing into R1 sub-region." [cite: 129] (R1 phía trên item/block đã đặt)
            # self._cutting_commands_current_layer.append(f"H_Cut y={y00+plh} for R1 above block (width {vbw})")
            self._execute_recursive_pack(vbw, vbh - plh, x00, y00 + plh)
        else: # dw is sufficient for other rectangles
            # Condition from Algorithm 1 [cite: 130]
            # (plw * (vbh - plh)) <= ((vbw - plw) * vbh) :
            # This compares potential area of R1 above item (width plw) vs R2 right of item (height vbh)
            if (plw * (vbh - plh)) <= ((vbw - plw) * vbh):
                # "placing into R1 sub-region." [cite: 130] (R1 phía trên, chiều rộng bằng item/block đã đặt - plw)
                # self._cutting_commands_current_layer.append(f"H_Cut y={y00+plh} for R1 above (width {plw})")
                # m1 = vbw (sao lưu vbw hiện tại)
                # self._execute_recursive_pack(plw, vbh - plh, x00, y00 + plh)
                #
                # "placing into R2 sub-region" [cite: 130] (R2 bên phải, chiều cao bằng vùng ban đầu - vbh)
                # self._cutting_commands_current_layer.append(f"V_Cut x={x00+plw} for R2 right (height {vbh})")
                # paper params: vbw=m1-plw; x00=plw+x00; vbh=plh+vbh (problematic); y00=y00-plh (problematic)
                # Corrected interpretation for R2: (vbw - plw, vbh, x00 + plw, y00)
                
                # The paper's Algorithm 1 structure for this branch:
                # pack R1 (width=plw, height=vbh-plh, pos=(x00, y00+plh))
                # pack R2 (width=vbw-plw, height=vbh, pos=(x00+plw, y00)) - this seems more logical.
                # The paper sets vbw=plw for R1 call, then vbw=m1-plw for R2.
                # This implies R1 has width plw.
                m1 = vbw # Store original vbw for this region before R1 call
                self._execute_recursive_pack(plw, vbh - plh, x00, y00 + plh) # R1 call with width plw
                self._execute_recursive_pack(m1 - plw, vbh, x00 + plw, y00) # R2 call with width m1-plw

            else: # Other branch for dividing space
                # "placing into R2 sub-region" [cite: 131] (R2 bên phải, chiều cao bằng item/block đã đặt - plh)
                # self._cutting_commands_current_layer.append(f"V_Cut x={x00+plw} for R2 right (height {plh})")
                # self._execute_recursive_pack(vbw - plw, plh, x00 + plw, y00)
                #
                # "placing into R1 sub-region." [cite: 131] (R1 phía trên, chiều rộng bằng vùng ban đầu - vbw)
                # self._cutting_commands_current_layer.append(f"H_Cut y={y00+plh} for R1 above (width {vbw})")
                # paper params: vbh=vbh-plh, vbw=plw (problematic); y00=plh+y00; x00=x00-plw (problematic)
                # Corrected interpretation for R1: (vbw, vbh - plh, x00, y00 + plh)
                
                # The paper's Algorithm 1 structure for this branch:
                # pack R2 (width=vbw-plw, height=plh, pos=(x00+plw, y00))
                # pack R1 (width=plw, height=vbh-plh, pos=(x00, y00+plh)) - this seems more logical for "R1"
                # The paper sets vbw=plw for R1 call here too.
                self._execute_recursive_pack(vbw - plw, plh, x00 + plw, y00) # R2 call
                self._execute_recursive_pack(plw, vbh - plh, x00, y00 + plh) # R1 call with width plw

    def run_packing_for_layer(self, target_layer_obj, Q_rects_for_this_layer):
        # Q_rects_for_this_layer: list of ORIGINAL Rectangle objects (or their deepcopies)
        # target_layer_obj: một Layer object đã có width, allocated_strip_height
        
        # Tạo bản đồ ID -> bản sao sâu của các hình chữ nhật để thao tác cục bộ
        self._Q_map = {r.id: deepcopy(r) for r in Q_rects_for_this_layer}
        for r_obj_copy in self._Q_map.values(): # Reset trạng thái is_placed cho lần xếp này
            r_obj_copy.is_placed = False
            r_obj_copy.x = None
            r_obj_copy.y = None

        self._placed_rects_current_layer = []
        self._cutting_commands_current_layer = []

        self._execute_recursive_pack(
            target_layer_obj.strip_width,
            target_layer_obj.allocated_strip_height,
            0, 0 # (x00, y00) - góc dưới trái của vùng ban đầu (toàn bộ lớp)
        )

        target_layer_obj.rectangles = self._placed_rects_current_layer # Gán các hình đã đặt
        target_layer_obj.cutting_commands = self._cutting_commands_current_layer
        target_layer_obj.calculate_content_dims_and_profit()

        # Dọn dẹp trạng thái nội bộ
        self._Q_map = {}
        self._placed_rects_current_layer = []
        self._cutting_commands_current_layer = []
        return target_layer_obj


# --- Thuật toán EDA chính (2DKEDA from Algorithm 2) ---
class TwoDKnapsackEDAAlgorithm:
    def __init__(self, all_rectangles_data, W_strip, H_strip, params):
        self.W_strip = W_strip # Chiều rộng dải chính [cite: 5]
        self.H_strip = H_strip # Chiều cao dải chính [cite: 5]
        self.params = params # Dictionary chứa tất cả các tham số điều khiển

        # Khởi tạo danh sách các hình chữ nhật gốc, bao gồm cả việc xây dựng khối ban đầu
        self.master_rectangles_list = self._initial_block_building_and_rect_list(
            all_rectangles_data, params.get('p_block_building_prob', 0.5)
        ) #
        self.n_total_master_rects = len(self.master_rectangles_list)

        # Các tham số quần thể và điều khiển vòng lặp
        self.population_size_initial_t = params.get('t_pop_initial', 5) # [cite: 196]
        self.population_size_max_tmax = params.get('tmax_pop_max', 30) # [cite: 195]
        self.current_population_size = self.population_size_initial_t

        self.kn_generations_loop = params.get('kn_loop_control', 5) # [cite: 196]
        self.time_end_limit = params.get('timeend_limit_seconds', 300) # [cite: 197] (hoặc số thế hệ)
        self.generations_count_limit = params.get('generations_limit', 200) # Giới hạn số thế hệ

        # Tham số mô hình xác suất
        self.alpha_learning_rate = params.get('alpha_learning', 0.2) # [cite: 155]
        self.prob_model_M1 = {r.id: 0.5 for r in self.master_rectangles_list} # [cite: 151, 199]
        self.prob_model_ECM = self._initialize_ECM_model() # [cite: 168, 199]

        # Tham số lựa chọn, đột biến, và lấy mẫu
        self.p_sampling_vs_mutation = params.get('p_samp_vs_mutate', 1.0) # [cite: 253]
        self.tp_truncation_selection_ratio = params.get('tp_selection_ratio', 0.1) # [cite: 254]

        # Tham số tìm kiếm cục bộ
        self.LSremn_iterations = params.get('LSremn_iters', 20) # [cite: 198] (phụ thuộc instance size)

        # Tham số khởi động lại (Restart)
        self.gp_restart_no_improvement_gens = params.get('gp_restart_gens', 300) # [cite: 197]
        self.rp_restart_delete_ratio = params.get('rp_restart_ratio', 0.7) # [cite: 197]
        self._generations_since_last_improvement = 0
        self._best_fitness_in_restart_period = -1.0

        # Các thành phần khác
        self.packer = PackLayerProcedure(params) # Đối tượng để thực hiện việc xếp lớp
        self.population = [] # Danh sách các SolutionIndividual
        self.best_solution_ever = None # Lưu trữ giải pháp tốt nhất tìm được

# Bên trong lớp TwoDKnapsackEDAAlgorithm (hoặc một tên tương tự bạn đã đặt)

    def _initial_block_building_and_rect_list(self, all_rectangles_data, p_block_prob): #
        """
        Xử lý dữ liệu đầu vào, tạo các Rectangle objects ban đầu.
        Sau đó, cố gắng xây dựng các khối (blocks) từ các hình chữ nhật cùng loại.
        Trả về một danh sách các Rectangle objects, bao gồm các hình chữ nhật gốc
        (chưa được dùng để tạo khối) và các khối đã được tạo.
        Mỗi item trong danh sách trả về phải có một ID duy nhất.
        """
        
        # Bước 1: Xử lý dữ liệu đầu vào và nhóm các hình chữ nhật theo type_id
        original_rects_by_type = {} # Sẽ chứa: {type_id_1: [rect_obj1, rect_obj2,...], type_id_2: [...]}
        temp_master_list = [] # Danh sách tạm thời chứa tất cả các hình chữ nhật gốc
        
        # Sử dụng self.W_strip và self.H_strip để kiểm tra xem khối có vừa không
        strip_W = self.W_strip
        strip_H = self.H_strip
        
        # current_id_val để đảm bảo ID là duy nhất cho cả hình gốc và khối
        # Chúng ta sẽ gán ID cuối cùng sau khi đã có danh sách hoàn chỉnh các item được chọn.
        # Tạm thời, các hình gốc có thể dùng ID từ dữ liệu hoặc một bộ đếm.
        
        # Tạo đối tượng Rectangle ban đầu và nhóm chúng
        temp_id_counter = 0
        for r_data in all_rectangles_data:
            type_id = r_data.get('type_id', temp_id_counter) # Nếu không có type_id, dùng id làm type_id
            rect = Rectangle(
                id=r_data.get('id', temp_id_counter), # Sử dụng ID từ input nếu có, hoặc tạm gán
                width=r_data['width'],
                height=r_data['height'],
                profit=r_data.get('profit'), # Sẽ tự tính nếu là None
                type_id=type_id,
                is_block_type=1 # btype=1: hình chữ nhật gốc [cite: 233]
            )
            temp_master_list.append(rect)
            if type_id not in original_rects_by_type:
                original_rects_by_type[type_id] = []
            original_rects_by_type[type_id].append(rect)
            temp_id_counter += 1

        # Bước 2: Danh sách kết quả cuối cùng sẽ chứa các hình gốc còn lại và các khối được tạo
        final_selectable_items = []
        
        # Biến đếm ID toàn cục mới cho các item trong final_selectable_items
        # để đảm bảo tính duy nhất sau khi block building
        final_item_id_counter = 0

        # Bước 3: Duyệt qua từng loại hình chữ nhật để thử tạo khối
        for type_id, rects_of_this_type in original_rects_by_type.items():
            # Làm việc trên một bản sao để có thể "tiêu thụ" hình chữ nhật khi tạo khối
            available_rects_for_blocking = deepcopy(rects_of_this_type) 
            
            # Lấy thông tin của hình chữ nhật mẫu (giả sử tất cả trong cùng type_id có kích thước/profit như nhau)
            if not available_rects_for_blocking:
                continue
            sample_rect = available_rects_for_blocking[0]
            orig_w, orig_h, orig_p = sample_rect.width, sample_rect.height, sample_rect.profit

            # "At every type with more elements the algorithm builds blocks with p_block probability." [cite: 230]
            # Giả sử "more elements" nghĩa là có đủ để tạo ít nhất một loại khối (ví dụ >1)
            if len(available_rects_for_blocking) > 1 and random.random() < p_block_prob:
                
                # 3.1. Thử tạo khối btype=4 (2x2)
                # "If the number of rectangles of the ith type is more than three..."
                # Nghĩa là cần ít nhất 4 hình để tạo một khối 2x2.
                while len(available_rects_for_blocking) >= 4:
                    block_w = 2 * orig_w
                    block_h = 2 * orig_h
                    # Kiểm tra xem khối có vừa với dải không
                    if block_w <= strip_W and block_h <= strip_H:
                        # Tạo khối
                        block_profit = 4 * orig_p # Giả sử profit cộng dồn (hoặc tính theo diện tích khối)
                        block_rect = Rectangle(id=final_item_id_counter, width=block_w, height=block_h,
                                            profit=block_profit, type_id=type_id, # Giữ type_id gốc hoặc tạo type_id mới cho khối
                                            is_block_type=4) # btype=4
                        final_selectable_items.append(block_rect)
                        final_item_id_counter += 1
                        # Tiêu thụ 4 hình chữ nhật đã dùng
                        available_rects_for_blocking = available_rects_for_blocking[4:]
                    else:
                        break # Khối không vừa, không thể tạo thêm khối 2x2 từ loại này

                # 3.2. Thử tạo khối btype=2 (2x1) hoặc btype=3 (1x2)
                # "If there are remained elements after the second step and the number of remained
                # rectangles of the ith type is more than two..."
                # Nghĩa là cần ít nhất 2 hình để tạo một khối 2x1 hoặc 1x2.
                while len(available_rects_for_blocking) >= 2:
                    chosen_btype = 0
                    block_w, block_h = 0, 0
                    
                    # "...with 0.5 probability it builds larger rectangles with 2*wj width and hj height
                    # (otherwise with wj width and 2*hj height)"
                    if random.random() < 0.5: # Thử khối 2x1 (ngang)
                        block_w = 2 * orig_w
                        block_h = orig_h
                        chosen_btype = 2 # btype=2
                    else: # Thử khối 1x2 (dọc)
                        block_w = orig_w
                        block_h = 2 * orig_h
                        chosen_btype = 3 # btype=3

                    if block_w <= strip_W and block_h <= strip_H:
                        block_profit = 2 * orig_p
                        block_rect = Rectangle(id=final_item_id_counter, width=block_w, height=block_h,
                                            profit=block_profit, type_id=type_id,
                                            is_block_type=chosen_btype)
                        final_selectable_items.append(block_rect)
                        final_item_id_counter += 1
                        available_rects_for_blocking = available_rects_for_blocking[2:]
                    else:
                        # Nếu loại khối này không vừa, thử loại khối kia (nếu có thể) hoặc dừng
                        # Để đơn giản, nếu một loại không vừa, ta dừng thử tạo khối 2-item cho type này.
                        # Một cách tiếp cận phức tạp hơn là thử loại khối 2-item còn lại.
                        break 
            
            # 3.3. Thêm các hình chữ nhật gốc còn lại (chưa được dùng để tạo khối) vào danh sách cuối cùng
            # Đây là các hình có btype=1
            for r_orig in available_rects_for_blocking:
                # Cần gán ID mới duy nhất cho chúng trong danh sách cuối cùng
                r_orig.id = final_item_id_counter # Gán lại ID để duy nhất trong final_selectable_items
                final_selectable_items.append(r_orig)
                final_item_id_counter += 1
                # r_orig.is_block_type vẫn là 1 (đã gán khi tạo)

        # "At the end the algorithm updates the number of the types and all rectangles get identification numbers." [cite: 238]
        # Việc cập nhật số loại hình chữ nhật (nếu khối được coi là loại mới) phức tạp hơn.
        # Ở đây, chúng ta chỉ đảm bảo tất cả các item trong final_selectable_items có ID duy nhất.
        # type_id của khối có thể giữ nguyên type_id của hình gốc để tham chiếu, hoặc bạn có thể tạo hệ thống type_id mới.

        return final_selectable_items

    def _initialize_ECM_model(self): # [cite: 168, 199]
        ecm = {}
        for i in range(len(self.master_rectangles_list)):
            for j in range(i + 1, len(self.master_rectangles_list)):
                r1_id = self.master_rectangles_list[i].id
                r2_id = self.master_rectangles_list[j].id
                ecm[tuple(sorted((r1_id, r2_id)))] = 0.5
        return ecm

# Bên trong lớp TwoDKnapsackEDAAlgorithm

    def _generate_initial_population(self): #
        self.population = []
        min_possible_layer_height_heuristic = 1 # Chiều cao tối thiểu để một lớp có ý nghĩa

        for i in range(self.population_size_initial_t):
            individual = SolutionIndividual(self.W_strip, self.H_strip)
            # Làm việc trên bản sao sâu của master list cho mỗi individual
            current_available_rects = deepcopy(self.master_rectangles_list)
            
            current_total_allocated_height = 0.0
            layer_id_counter = 0
            first_layer_determined_height = 0.0 # Sẽ lưu chiều cao ngẫu nhiên của lớp đầu tiên

            while current_available_rects and current_total_allocated_height < self.H_strip:
                allocated_h_for_this_layer = 0.0

                # Ước tính chiều cao tối thiểu cần thiết cho các hình còn lại
                min_h_of_remaining_rects = float('inf')
                if current_available_rects:
                    min_h_of_remaining_rects = min(r.height for r in current_available_rects)
                
                # Không thể tạo lớp nếu chiều cao còn lại quá nhỏ cho bất kỳ hình nào
                if self.H_strip - current_total_allocated_height < min_h_of_remaining_rects and current_available_rects:
                    break


                if not individual.layers: # Đây là lớp đầu tiên
                    min_h_rand = self.H_strip * 0.3 #
                    max_h_rand = self.H_strip - 1    #
                    
                    # Đảm bảo khoảng hợp lệ và không vượt quá chiều cao strip còn lại
                    min_h_rand = max(min_possible_layer_height_heuristic, min_h_rand)
                    max_h_rand = max(min_h_rand, max_h_rand) # Đảm bảo max_h >= min_h
                    
                    potential_h = random.uniform(min_h_rand, max_h_rand) if min_h_rand < max_h_rand else min_h_rand
                    allocated_h_for_this_layer = min(potential_h, self.H_strip - current_total_allocated_height)
                    first_layer_determined_height = allocated_h_for_this_layer # Lưu lại chiều cao này
                else: # Các lớp tiếp theo
                    # "pack the next layer with the same height"
                    if first_layer_determined_height > 0:
                        allocated_h_for_this_layer = min(first_layer_determined_height, 
                                                        self.H_strip - current_total_allocated_height)
                    else: 
                        # Fallback nếu first_layer_determined_height không hợp lệ (ví dụ =0)
                        # Chọn một phần ngẫu nhiên của chiều cao còn lại
                        remaining_strip_h = self.H_strip - current_total_allocated_height
                        allocated_h_for_this_layer = random.uniform(min_possible_layer_height_heuristic, remaining_strip_h) if remaining_strip_h > min_possible_layer_height_heuristic else remaining_strip_h


                # Đảm bảo chiều cao lớp đủ lớn và không âm
                allocated_h_for_this_layer = max(min_possible_layer_height_heuristic, allocated_h_for_this_layer)
                if allocated_h_for_this_layer <= 0.01: # Quá nhỏ để có ý nghĩa
                    break
                
                # Tạo shell cho lớp mới
                new_layer_obj = Layer(layer_id=layer_id_counter, 
                                    strip_width=self.W_strip, 
                                    allocated_strip_height=allocated_h_for_this_layer)
                
                # Thực hiện xếp hình cho lớp này
                # self.packer.run_packing_for_layer nhận list các rects để thử xếp
                # và sẽ cập nhật new_layer_obj.rectangles với những hình đã đặt thành công
                # (các hình trong new_layer_obj.rectangles là bản sao đã được cập nhật is_placed, x, y)
                self.packer.run_packing_for_layer(new_layer_obj, current_available_rects)

                if new_layer_obj.rectangles: # Nếu có ít nhất một hình chữ nhật được xếp vào lớp
                    individual.layers.append(new_layer_obj)
                    current_total_allocated_height += new_layer_obj.allocated_strip_height
                    layer_id_counter += 1

                    # Cập nhật danh sách current_available_rects
                    placed_ids_in_this_layer = {r.id for r in new_layer_obj.rectangles}
                    current_available_rects = [r for r in current_available_rects if r.id not in placed_ids_in_this_layer]
                else:
                    # Không có hình nào được xếp vào lớp này (có thể do chiều cao cấp phát quá nhỏ
                    # hoặc không còn hình nào phù hợp), dừng việc thêm lớp.
                    break
            
            # Các hình chữ nhật còn lại trong current_available_rects sẽ vào rem_set
            individual.rem_set = current_available_rects #
            individual.calculate_fitness_and_total_height()

            # "If it is necessary at end it applies the repair procedure."
            if not individual.is_feasible(): # total_allocated_height > H_strip
                self._repair_procedure(individual) # Placeholder, sẽ được triển khai sau
                individual.calculate_fitness_and_total_height() # Tính lại sau khi sửa chữa

            self.population.append(individual)

        # Kết thúc vòng lặp tạo quần thể
        if self.population:
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            self.best_solution_ever = deepcopy(self.population[0])
            self._best_fitness_in_restart_period = self.best_solution_ever.fitness
        else:
            # Xử lý trường hợp không tạo được cá thể nào (rất hiếm nếu có hình chữ nhật)
            print("Warning: Initial population is empty.")

    def _update_M1_model(self): #
        if not self.population:
            print("Warning: Population is empty. Cannot update M1 model.")
            return

        # 1. Lấy 20% cá thể tốt nhất từ self.population
        num_total_individuals = len(self.population)
        num_best_to_consider = max(1, int(0.2 * num_total_individuals)) # [cite: 156] (đảm bảo ít nhất 1)
        best_individuals = self.population[:num_best_to_consider] # Giả định population đã được sắp xếp

        if not best_individuals: # Trường hợp hiếm gặp nếu num_best_to_consider = 0
            return

        # 2. Khởi tạo delta_M1_counts
        # self.prob_model_M1 đã được khởi tạo với tất cả các rect_id từ master_rectangles_list
        delta_M1_counts = {rect_id: 0 for rect_id in self.prob_model_M1.keys()}

        # 3. Đếm số lần mỗi hình chữ nhật xuất hiện trong các giải pháp tốt nhất
        for individual in best_individuals:
            placed_rect_ids_in_individual = individual.get_all_placed_rect_ids() # Lấy ID các hình đã đặt
            for rect_id in placed_rect_ids_in_individual:
                if rect_id in delta_M1_counts: # Đảm bảo rect_id có trong model
                    delta_M1_counts[rect_id] += 1 # [cite: 157]
                # else:
                # print(f"Warning: rect_id {rect_id} from individual not in master M1 model keys.")


        # 4. Tính delta_M1_relative_freq (ΔM1_i)
        delta_M1_relative_freq = {}
        for rect_id, count in delta_M1_counts.items():
            delta_M1_relative_freq[rect_id] = count / num_best_to_consider # [cite: 158]

        # 5. Cập nhật self.prob_model_M1 theo công thức
        for rect_id in self.prob_model_M1.keys():
            current_m1_value = self.prob_model_M1[rect_id]
            delta_m1_for_id = delta_M1_relative_freq.get(rect_id, 0.0) # Lấy tần suất, nếu không có thì là 0

            # Áp dụng công thức: M1_i^(gen+1) = (1-alpha)M1_i^gen + alpha*delta_M1_i [cite: 155]
            self.prob_model_M1[rect_id] = \
                (1 - self.alpha_learning_rate) * current_m1_value + \
                self.alpha_learning_rate * delta_m1_for_id
                
            # Đảm bảo xác suất nằm trong khoảng [0, 1] (tùy chọn, nhưng nên có nếu có sai số float)
            self.prob_model_M1[rect_id] = max(0.0, min(1.0, self.prob_model_M1[rect_id]))

        # print(f"Framework: M1 model updated. Example M1 value for rect ID 0: {self.prob_model_M1.get(0, 'N/A')}")

    def _update_ECM_model(self): #
        if not self.population:
            print("Warning: Population is empty. Cannot update ECM model.")
            return

        # 1. Lấy 20% cá thể tốt nhất
        num_total_individuals = len(self.population)
        num_best_to_consider = max(1, int(0.2 * num_total_individuals))
        best_individuals = self.population[:num_best_to_consider] # Giả định population đã được sắp xếp

        if not best_individuals:
            return

        # 2. Khởi tạo delta_ECM_counts
        # self.prob_model_ECM đã được khởi tạo với tất cả các cặp (id1, id2) với id1 < id2
        delta_ECM_counts = {pair: 0 for pair in self.prob_model_ECM.keys()}

        # 3. Đếm số lần mỗi cặp hình chữ nhật xuất hiện cùng nhau trong một lớp
        for individual in best_individuals: # [cite: 173]
            for layer_obj in individual.layers:
                ids_in_layer = sorted([rect.id for rect in layer_obj.rectangles if rect.is_placed]) # Sắp xếp để tạo cặp chuẩn hóa
                
                # Tạo tất cả các cặp có thể từ các ID trong lớp này
                for i in range(len(ids_in_layer)):
                    for j in range(i + 1, len(ids_in_layer)):
                        # Cặp luôn được lưu với id nhỏ hơn ở trước
                        pair = (ids_in_layer[i], ids_in_layer[j]) 
                        if pair in delta_ECM_counts:
                            delta_ECM_counts[pair] += 1
                        # else:
                            # print(f"Warning: Pair {pair} from individual's layer not in master ECM model keys.")

        # 4. Tính delta_ECM_relative_freq (ΔECM_ij)
        delta_ECM_relative_freq = {}
        for pair, count in delta_ECM_counts.items():
            delta_ECM_relative_freq[pair] = count / num_best_to_consider # (chia cho số lượng 'best' individuals)

        # 5. Cập nhật self.prob_model_ECM theo công thức
        for pair in self.prob_model_ECM.keys():
            current_ecm_value = self.prob_model_ECM[pair]
            delta_ecm_for_pair = delta_ECM_relative_freq.get(pair, 0.0)

            # Áp dụng công thức: ECM_ij^(gen+1) = (1-alpha)ECM_ij^gen + alpha*delta_ECM_ij [cite: 172]
            self.prob_model_ECM[pair] = \
                (1 - self.alpha_learning_rate) * current_ecm_value + \
                self.alpha_learning_rate * delta_ecm_for_pair
                
            # Đảm bảo xác suất nằm trong khoảng [0, 1] (tùy chọn)
            self.prob_model_ECM[pair] = max(0.0, min(1.0, self.prob_model_ECM[pair]))
        
        # print(f"Framework: ECM model updated. Example ECM value for a pair: {next(iter(self.prob_model_ECM.items())) if self.prob_model_ECM else 'N/A'}")


    def _sample_M1_for_knapsack(self): #
        """
        Chọn một tập con các hình chữ nhật QK từ self.master_rectangles_list
        dựa trên xác suất trong self.prob_model_M1.
        Trả về một list các Rectangle objects (là bản sao sâu).
        """
        QK_selected_rect_ids = set() # Sử dụng set để tránh trùng lặp ID nếu có lỗi logic nào đó

        # Duyệt qua từng hình chữ nhật trong danh sách tổng thể
        for r_master in self.master_rectangles_list:
            # Lấy xác suất chọn hình chữ nhật này từ mô hình M1
            # Mặc định là 0.0 nếu ID không có trong mô hình (dù điều này không nên xảy ra nếu M1 được khởi tạo đúng)
            probability_of_selection = self.prob_model_M1.get(r_master.id, 0.0)

            # "It selects the rectangles where a random probability < M1_i^gen" [cite: 160]
            if random.random() < probability_of_selection:
                QK_selected_rect_ids.add(r_master.id)

        # Tạo danh sách các đối tượng Rectangle (bản sao sâu) từ các ID đã chọn
        QK_rect_objects = []
        for r_master in self.master_rectangles_list:
            if r_master.id in QK_selected_rect_ids:
                QK_rect_objects.append(deepcopy(r_master))
                # Đảm bảo các bản sao này có trạng thái is_placed là False ban đầu
                QK_rect_objects[-1].is_placed = False
                QK_rect_objects[-1].x = None
                QK_rect_objects[-1].y = None
                QK_rect_objects[-1].x_global = None
                QK_rect_objects[-1].y_global = None


        # print(f"Framework: Sampled M1 for knapsack (QK). Selected {len(QK_rect_objects)} rectangles.")
        return QK_rect_objects

    def _calculate_Pr_ij_for_ECM_sampling(self, rect_i_id, rect_j_id): #
        # Tính Pr_ij = ECM_ij^gen / sum_over_k(ECM_ik^gen)
        # ECM_ij là giá trị hiện tại trong self.prob_model_ECM
        # sum_over_k(ECM_ik) là tổng các giá trị ECM của rect_i với tất cả các rect_k khác.
        
        pair_ij_key = tuple(sorted((rect_i_id, rect_j_id)))
        ecm_ij_value = self.prob_model_ECM.get(pair_ij_key, 0.0)

        sum_ecm_ik_values = 1e-9 # Để tránh chia cho 0
        for r_k in self.master_rectangles_list:
            if r_k.id == rect_i_id: # Bỏ qua chính nó
                continue
            # if r_k.id == rect_j_id: # Đã được tính trong ecm_ij_value nếu k=j
            #     sum_ecm_ik_values += ecm_ij_value 
            # else:
            pair_ik_key = tuple(sorted((rect_i_id, r_k.id)))
            sum_ecm_ik_values += self.prob_model_ECM.get(pair_ik_key, 0.0)
                
        if sum_ecm_ik_values < 1e-8 : # Gần như bằng 0
            return 0.0 
        return ecm_ij_value / sum_ecm_ik_values


    def _sample_ECM_for_layers_and_pack(self, QK_rects_list, target_individual): #
        target_individual.layers = [] # Xóa các lớp cũ nếu có
        
        qk_ids_set = {r.id for r in QK_rects_list}
        target_individual.rem_set = [deepcopy(r) for r in self.master_rectangles_list if r.id not in qk_ids_set]

        unselected_QK_rects = deepcopy(QK_rects_list) # Làm việc trên bản sao
        
        current_total_allocated_height = 0.0
        layer_id_counter = 0
        min_practical_layer_height = 1.0 # Ngưỡng chiều cao tối thiểu cho một lớp

        # "1. If there are unselected rectangles in QK, first it chooses an unselected rectangle randomly from QK;" [cite: 179]
        #    "...if there is not, go to step 3." [cite: 180] (Step 3 là kết thúc lấy mẫu)
        while unselected_QK_rects and current_total_allocated_height < self.H_strip:
            first_rect_for_layer = random.choice(unselected_QK_rects)
            unselected_QK_rects.remove(first_rect_for_layer) # Loại bỏ khỏi danh sách chờ

            Q_for_this_layer = [first_rect_for_layer] # Hình chữ nhật đầu tiên cho lớp

            # "For the other rectangles of Q the sampling selects the rectangles from QK, where
            # a random probability < pr_ij (the ith and jth rectangles from QK)." [cite: 181]
            
            # Tạo bản sao để duyệt vì unselected_QK_rects sẽ bị thay đổi
            candidates_for_current_layer = unselected_QK_rects[:] 
            for r_candidate in candidates_for_current_layer:
                # pr_ij là xác suất r_candidate tạo cặp tốt với first_rect_for_layer
                # Sử dụng hàm đã tạo ở trên
                prob_good_pair = self._calculate_Pr_ij_for_ECM_sampling(first_rect_for_layer.id, r_candidate.id)
                if random.random() < prob_good_pair:
                    Q_for_this_layer.append(r_candidate)
                    if r_candidate in unselected_QK_rects: # Kiểm tra trước khi xóa
                        unselected_QK_rects.remove(r_candidate)

            # "If Q is ready then it calls the packlayer procedure..." [cite: 182]
            if not Q_for_this_layer: # Không có hình nào cho lớp này (hiếm)
                continue

            # Xác định chiều cao cấp phát cho lớp này
            min_h_needed_for_Q = min(r.height for r in Q_for_this_layer) if Q_for_this_layer else min_practical_layer_height
            
            # Chiều cao còn lại trên dải
            remaining_strip_height = self.H_strip - current_total_allocated_height
            
            if remaining_strip_height < min_h_needed_for_Q: # Không đủ chiều cao còn lại
                # Các hình trong Q_for_this_layer không thể tạo lớp, trả chúng vào unselected_QK_rects
                # để sau này được thêm vào rem_set của individual
                unselected_QK_rects.extend(Q_for_this_layer)
                break # Dừng tạo lớp

            # Cấp phát một phần hoặc toàn bộ chiều cao còn lại một cách hợp lý
            # Ví dụ: lấy một nửa chiều cao còn lại, nhưng không nhỏ hơn min_h_needed_for_Q
            allocated_h_for_this_layer = max(min_h_needed_for_Q, random.uniform(min_h_needed_for_Q, remaining_strip_height))
            allocated_h_for_this_layer = min(allocated_h_for_this_layer, remaining_strip_height) # Đảm bảo không vượt quá
            allocated_h_for_this_layer = max(allocated_h_for_this_layer, min_practical_layer_height) # Đảm bảo có ý nghĩa

            if allocated_h_for_this_layer <=0: break


            current_layer_obj = Layer(layer_id=layer_id_counter, 
                                    strip_width=self.W_strip, 
                                    allocated_strip_height=allocated_h_for_this_layer)
            
            # Gọi packlayer
            self.packer.run_packing_for_layer(current_layer_obj, Q_for_this_layer)

            # "...the Repacking procedure..." [cite: 182]
            self._repacking_local_search(current_layer_obj) # Placeholder

            # "...and after the profitrepair local search procedure for the layer in this order." [cite: 182]
            # "The profitrepair procedure selects all the unselected rectangles of QK and all the rectangles of rem..." [cite: 183]
            candidates_for_profit_repair = unselected_QK_rects + target_individual.rem_set
            self._profit_repair_local_search(current_layer_obj, candidates_for_profit_repair) # Placeholder

            if current_layer_obj.rectangles: # Nếu lớp có chứa hình chữ nhật sau các bước trên
                target_individual.layers.append(current_layer_obj)
                current_total_allocated_height += current_layer_obj.allocated_strip_height
                layer_id_counter += 1
            else: # Lớp rỗng, các hình chữ nhật trong Q_for_this_layer không được xếp
                target_individual.rem_set.extend(Q_for_this_layer) # Thêm chúng vào rem_set chung

        # Kết thúc vòng lặp while tạo lớp
        # "If there are unselected rectangles in QK we repeat the layer generation..." [cite: 186] (đã xử lý bởi while)
        
        # Thêm các hình chữ nhật QK còn lại (nếu có) vào rem_set của individual
        target_individual.rem_set.extend(unselected_QK_rects)
        # Loại bỏ các bản sao trong rem_set nếu có
        unique_rem_set_ids = set()
        unique_rem_set_rects = []
        for r_rem in target_individual.rem_set:
            if r_rem.id not in unique_rem_set_ids:
                unique_rem_set_rects.append(r_rem)
                unique_rem_set_ids.add(r_rem.id)
        target_individual.rem_set = unique_rem_set_rects

        target_individual.calculate_fitness_and_total_height()

        # "If the total height of the layers is higher as H, it calls the repair procedure." [cite: 187]
        if not target_individual.is_feasible():
            self._repair_procedure(target_individual) # Placeholder
            target_individual.calculate_fitness_and_total_height() # Tính lại sau khi sửa chữa

# Bên trong lớp TwoDKnapsackEDAAlgorithm (hoặc tên tương tự bạn đã đặt)

    def _generate_descendant_by_sampling(self): # [cite: 189] (implicit), [cite: 201] (local search part)
        """
        Tạo một SolutionIndividual mới bằng cách lấy mẫu M1 và ECM,
        sau đó áp dụng các tìm kiếm cục bộ chính.
        """
        # print("Framework: Generating descendant by M1/ECM sampling...")
        new_individual = SolutionIndividual(self.W_strip, self.H_strip)

        # "Generate the rectangles for the knapsack by sampling M1."
        qk_rects_list = self._sample_M1_for_knapsack() #

        # "Generate the layers by sampling ECM."
        # Phương thức này đã bao gồm việc gọi packer, RepackingLS, ProfitRepairLS như mô tả trong mục 4.2
        self._sample_ECM_for_layers_and_pack(qk_rects_list, new_individual) #

        # "Apply local searches." (Áp dụng LSrem1, LSrem2, LSrem3)
        # Giả sử _apply_main_local_searches sẽ gọi nhóm LSrem được mô tả trong mục 5.2
        self._apply_main_local_searches(new_individual) #

        # Tính toán lại fitness và các thuộc tính khác sau khi tất cả các thay đổi đã được thực hiện
        new_individual.calculate_fitness_and_total_height()
        
        return new_individual

    def _truncation_selection(self): #
        """
        Chọn một individual từ self.population dựa trên lựa chọn cắt cụt.
        Chỉ xem xét tp_truncation_selection_ratio (ví dụ 10%) cá thể tốt nhất.
        Chọn ngẫu nhiên một cá thể từ nhóm này.
        Trả về một bản sao sâu (deepcopy) của individual được chọn.
        """
        # print("Framework: Performing truncation selection...")
        if not self.population:
            # print("Warning: Population is empty, cannot perform truncation selection.")
            return None

        # Xác định số lượng cá thể tốt nhất để xem xét
        # self.tp_truncation_selection_ratio là một tỷ lệ, ví dụ 0.1 cho 10%
        num_to_consider = max(1, int(self.tp_truncation_selection_ratio * len(self.population)))
        
        # Lấy ra các cá thể tốt nhất (giả sử self.population đã được sắp xếp theo fitness giảm dần)
        eligible_parents_pool = self.population[:num_to_consider]

        if not eligible_parents_pool:
            # Trường hợp hiếm nếu num_to_consider = 0 hoặc quần thể rất nhỏ
            # print("Warning: No eligible parents in truncation selection pool.")
            return None

        # Chọn ngẫu nhiên một cha mẹ từ nhóm các cá thể tốt nhất
        selected_parent = random.choice(eligible_parents_pool)
        
        return deepcopy(selected_parent) # Trả về bản sao sâu

    def _find_rect_pair_with_max_pr_in_layer(self, layer_rects):
        """Helper: Tìm cặp (i,j) trong layer_rects có Pr_ij lớn nhất."""
        max_pr = -1.0
        best_pair_rects = (None, None)
        if len(layer_rects) < 2:
            return best_pair_rects

        for idx1 in range(len(layer_rects)):
            for idx2 in range(idx1 + 1, len(layer_rects)):
                r1 = layer_rects[idx1]
                r2 = layer_rects[idx2]
                pr_val = self._calculate_Pr_ij_for_ECM_sampling(r1.id, r2.id)
                if pr_val > max_pr:
                    max_pr = pr_val
                    best_pair_rects = (r1, r2)
        return best_pair_rects

    def _find_rect_with_max_pr_to_external_rect_in_layer(self, layer_rects, external_rect_id):
        """Helper: Tìm rect z trong layer_rects có Pr_{external_rect_id, z} lớn nhất."""
        max_pr = -1.0
        best_rect_z = None
        if not layer_rects:
            return None
            
        for r_z_candidate in layer_rects:
            pr_val = self._calculate_Pr_ij_for_ECM_sampling(external_rect_id, r_z_candidate.id)
            if pr_val > max_pr:
                max_pr = pr_val
                best_rect_z = r_z_candidate
        return best_rect_z

    def _are_rects_in_same_layer_in_solution(self, rect_id1, rect_id2, solution_individual):
        """Helper: Kiểm tra xem rect_id1 và rect_id2 có cùng lớp trong solution_individual không."""
        if not solution_individual: return False
        for layer in solution_individual.layers:
            ids_in_layer = {r.id for r in layer.rectangles}
            if rect_id1 in ids_in_layer and rect_id2 in ids_in_layer:
                return True
        return False

    def _mutation_based_on_ECM(self, individual_to_mutate: SolutionIndividual): #
        # print(f"Framework: Applying ECM-based mutation to individual (ID hash: {hash(str(individual_to_mutate.layers))})...")
        
        if len(individual_to_mutate.layers) < 2:
            # print("    Mutation_ECM: Not enough layers for mutation (<2).")
            return

        modified_layers_indices = set() # Theo dõi các lớp đã bị thay đổi

        # "It repeats the swaps three times" [cite: 256]
        for swap_attempt_num in range(3):
            # === Bước a: Chọn layer1 và các hình chữ nhật i, j, k ===
            eligible_layers_for_l1 = []
            for idx, l_obj in enumerate(individual_to_mutate.layers):
                if len(l_obj.rectangles) >= 3: # "...number of the rectangles in the layer >=3" [cite: 248]
                    eligible_layers_for_l1.append((idx, l_obj))
            
            if not eligible_layers_for_l1:
                # print(f"    Mutation_ECM attempt {swap_attempt_num+1}: No eligible layer1 (need >=3 rects). Skipping this swap attempt.")
                continue 
            
            l1_idx, layer1 = random.choice(eligible_layers_for_l1)
            
            # "selects the ith and the jth rectangles from the given layer with the largest pr_ij" [cite: 248]
            rect_i, rect_j = self._find_rect_pair_with_max_pr_in_layer(layer1.rectangles)
            if not rect_i or not rect_j: # Không tìm được cặp i,j (ví dụ lớp chỉ còn <2 hình sau khi các hình khác bị xóa)
                # print(f"    Mutation_ECM attempt {swap_attempt_num+1}: Could not find rect_i, rect_j in layer {l1_idx}. Skipping.")
                continue

            # "selects randomly another kth rectangles from the layer" [cite: 248]
            potential_k_rects = [r for r in layer1.rectangles if r.id != rect_i.id and r.id != rect_j.id]
            if not potential_k_rects:
                # print(f"    Mutation_ECM attempt {swap_attempt_num+1}: No potential rect_k in layer {l1_idx}. Skipping.")
                continue
            rect_k = random.choice(potential_k_rects)

            # === Bước b: Kiểm tra điều kiện với cá thể tốt nhất ===
            # "If the ith and kth rectangles are in the same layer of the best individual it do not make swap." [cite: 249]
            if self.best_solution_ever and \
            self._are_rects_in_same_layer_in_solution(rect_i.id, rect_k.id, self.best_solution_ever):
                # print(f"    Mutation_ECM attempt {swap_attempt_num+1}: R{rect_i.id} and R{rect_k.id} in same layer of best solution. Skipping swap.")
                continue
                
            # === Bước c: Chọn layer2 và hình chữ nhật z ===
            # "Otherwise it chooses the zth rectangle from other layer with the largest pr_iz probability" [cite: 250]
            eligible_layers_for_l2 = []
            for idx, l_obj in enumerate(individual_to_mutate.layers):
                if idx != l1_idx and len(l_obj.rectangles) >= 1:
                    eligible_layers_for_l2.append((idx, l_obj))

            if not eligible_layers_for_l2:
                # print(f"    Mutation_ECM attempt {swap_attempt_num+1}: No eligible layer2 (need >=1 rect, different from L1). Skipping.")
                continue
            
            best_rect_z_overall = None
            best_pr_iz_overall = -1.0
            selected_l2_idx = -1
            final_layer2_obj = None

            for l2_idx_candidate, layer2_candidate_obj in eligible_layers_for_l2:
                # Tìm rect_z trong layer2_candidate_obj có pr_iz lớn nhất với rect_i
                rect_z_in_this_l2 = self._find_rect_with_max_pr_to_external_rect_in_layer(
                    layer2_candidate_obj.rectangles, rect_i.id
                )
                if rect_z_in_this_l2:
                    current_pr_iz = self._calculate_Pr_ij_for_ECM_sampling(rect_i.id, rect_z_in_this_l2.id)
                    if current_pr_iz > best_pr_iz_overall:
                        best_pr_iz_overall = current_pr_iz
                        best_rect_z_overall = rect_z_in_this_l2
                        selected_l2_idx = l2_idx_candidate
                        final_layer2_obj = layer2_candidate_obj
            
            if not best_rect_z_overall or final_layer2_obj is None:
                # print(f"    Mutation_ECM attempt {swap_attempt_num+1}: Could not find suitable rect_z. Skipping.")
                continue
                
            rect_z = best_rect_z_overall # Đây là hình chữ nhật z cuối cùng được chọn
            layer2 = final_layer2_obj   # Đây là lớp 2 cuối cùng được chọn

            # === Bước d: Thực hiện hoán đổi k (từ layer1) và z (từ layer2) ===
            try:
                # Quan trọng: làm việc với các đối tượng Rectangle thực sự trong danh sách
                # Cần tìm đúng đối tượng rect_k trong layer1.rectangles và rect_z trong layer2.rectangles
                # Giả định rằng rect_k và rect_z là các tham chiếu đến các đối tượng trong danh sách
                
                # Tạo bản sao trước khi thêm vào lớp khác để tránh lỗi tham chiếu nếu k và z là cùng một đối tượng
                # (mặc dù logic chọn đảm bảo chúng từ các lớp khác nhau, nhưng cẩn thận với ID)
                rect_k_copy_for_l2 = deepcopy(rect_k)
                rect_z_copy_for_l1 = deepcopy(rect_z)

                layer1.rectangles.remove(rect_k) 
                layer2.rectangles.remove(rect_z) 
                
                layer1.rectangles.append(rect_z_copy_for_l1)
                layer2.rectangles.append(rect_k_copy_for_l2)
                
                modified_layers_indices.add(l1_idx)
                modified_layers_indices.add(selected_l2_idx)
                # print(f"    Mutation_ECM attempt {swap_attempt_num+1}: Swapped R{rect_k.id}(L{l1_idx}) with R{rect_z.id}(L{selected_l2_idx}).")

            except ValueError: # Nếu rect_k hoặc rect_z không tìm thấy (lỗi logic hiếm gặp)
                # print(f"    Mutation_ECM attempt {swap_attempt_num+1}: Error removing rect_k or rect_z. Skipping swap.")
                # Có thể cần khôi phục lại layer1.rectangles và layer2.rectangles nếu một phần remove thành công
                continue

        # Kết thúc vòng lặp 3 lần hoán đổi

        # "and after applies packlayer, repacking for the layers." [cite: 256]
        if modified_layers_indices:
            # print(f"    Mutation_ECM: Repacking modified layers: {list(modified_layers_indices)}")
            for layer_idx_to_repack in modified_layers_indices:
                layer_to_repack = individual_to_mutate.layers[layer_idx_to_repack]
                
                # Lấy danh sách các hình chữ nhật hiện tại của lớp để xếp lại
                # Cần bản sao sâu và reset trạng thái is_placed
                current_rects_in_layer_for_repack = [deepcopy(r) for r in layer_to_repack.rectangles]
                if not current_rects_in_layer_for_repack: # Nếu lớp trở nên rỗng sau khi swap
                    layer_to_repack.rectangles = []
                    layer_to_repack.cutting_commands = []
                    layer_to_repack.calculate_content_dims_and_profit()
                    layer_to_repack.allocated_strip_height = layer_to_repack.content_actual_height # Có thể bằng 0
                    continue

                for r_copy in current_rects_in_layer_for_repack:
                    r_copy.is_placed = False; r_copy.x=None; r_copy.y=None
                
                # Khi gọi lại packlayer, sử dụng allocated_strip_height hiện tại của lớp làm giới hạn
                # self.packer.run_packing_for_layer sẽ cập nhật nội dung của layer_to_repack
                self.packer.run_packing_for_layer(layer_to_repack, current_rects_in_layer_for_repack)
                
                # Sau đó áp dụng repacking
                self._repacking_local_search(layer_to_repack) # Placeholder

        # "If the total height of the layers is too large, it applies the repair procedure." [cite: 257]
        individual_to_mutate.calculate_fitness_and_total_height() # Tính tổng chiều cao và fitness mới
        if not individual_to_mutate.is_feasible():
            # print(f"    Mutation_ECM: Total height {individual_to_mutate.total_allocated_height:.2f} > {self.H_strip:.2f}. Applying repair.")
            self._repair_procedure(individual_to_mutate) # Placeholder
            individual_to_mutate.calculate_fitness_and_total_height() # Tính lại sau repair

        # print(f"    Mutation_ECM: Finished. Individual new fitness: {individual_to_mutate.fitness:.2f}")

    def _generate_descendant_by_mutation(self): # (implicit, local search part)
        """
        Tạo một hậu duệ mới bằng cách chọn lọc cha, đột biến dựa trên ECM,
        và sau đó áp dụng các tìm kiếm cục bộ chính.
        """
        # print("Framework: Generating descendant by selection & mutation...")
        
        # 1. Chọn cha bằng lựa chọn cắt cụt
        parent_individual = self._truncation_selection() # [cite: 254]

        # Nếu không chọn được cha (ví dụ, quần thể rỗng hoặc quá nhỏ),
        # quay lại tạo hậu duệ bằng phương pháp lấy mẫu
        if parent_individual is None:
            # print("Warning: No parent selected for mutation, falling back to sampling method.")
            return self._generate_descendant_by_sampling() 

        # 2. Tạo bản sao của cha để làm hậu duệ
        # Hàm _truncation_selection đã trả về một deepcopy
        descendant_individual = parent_individual 

        # 3. Áp dụng đột biến dựa trên ECM cho hậu duệ
        self._mutation_based_on_ECM(descendant_individual) #
        # Hàm _mutation_based_on_ECM nên bao gồm việc gọi packlayer, repacking, và repair nếu cần,
        # như mô tả trong bài báo.

        # 4. Áp dụng các tìm kiếm cục bộ chính (LSrem1, LSrem2, LSrem3)
        # Giống như sau khi tạo hậu duệ bằng cách lấy mẫu
        self._apply_main_local_searches(descendant_individual) #

        # 5. Tính toán lại fitness và các thuộc tính khác của hậu duệ
        descendant_individual.calculate_fitness_and_total_height()
        
        return descendant_individual

    # --- Các thủ tục Tìm kiếm Cục bộ (Local Search) ---
    def _repacking_local_search(self, layer_obj: Layer): #
        """
        Cố gắng giảm chiều cao (allocated_strip_height) của layer_obj bằng cách
        xếp lại các hình chữ nhật hiện có của nó vào một không gian vừa khít hơn.
        Sửa đổi trực tiếp layer_obj.
        """
        # print(f"Framework: LS - Repacking layer {layer_obj.id} (current allocated H: {layer_obj.allocated_strip_height:.2f}, content H: {layer_obj.content_actual_height:.2f})...")

        if not layer_obj.rectangles:
            layer_obj.content_actual_height = 0
            layer_obj.allocated_strip_height = 0 # Không có gì, không cần chiều cao
            layer_obj.total_profit = 0
            # print(f"    Layer {layer_obj.id} is empty. Repacked height set to 0.")
            return

        # 1. Lấy danh sách hình chữ nhật hiện tại để xếp lại
        # Tạo bản sao sâu để không làm xáo trộn các thuộc tính (như x,y) của các hình gốc trong layer_obj
        # trong trường hợp packer_procedure sửa đổi chúng trực tiếp (mặc dù packer của chúng ta làm việc với bản sao trong _Q_map)
        rects_to_repack_copies = [deepcopy(r) for r in layer_obj.rectangles]
        
        # Reset trạng thái is_placed, x, y cho các bản sao này trước khi xếp lại
        for r_copy in rects_to_repack_copies:
            r_copy.is_placed = False
            r_copy.x = None
            r_copy.y = None
            # Các thuộc tính khác như width, height, profit, id, type_id được giữ nguyên

        # 2. Xác định chiều cao mục tiêu cho việc xếp lại
        # Một chiến lược là thử xếp vào chiều cao nội dung thực tế hiện tại của lớp.
        # Nếu packer tốt, nó có thể tìm ra cách xếp còn tốt hơn (content_actual_height thấp hơn).
        # Hoặc, nếu lớp ban đầu có allocated_strip_height > content_actual_height,
        # thì việc xếp lại vào content_actual_height là một cách để "siết chặt".
        
        # Tính chiều cao nội dung hiện tại nếu nó chưa được cập nhật gần đây
        current_content_h, _ = layer_obj.calculate_content_dims_and_profit() # Đảm bảo content_actual_height là mới nhất

        # Chiều cao mục tiêu để thử xếp lại. Ít nhất phải bằng chiều cao của hình cao nhất.
        if not rects_to_repack_copies: # Double check, dù đã kiểm tra ở trên
            target_repack_height = 0
        else:
            max_individual_rect_height = max(r.height for r in rects_to_repack_copies) if rects_to_repack_copies else 0
            # Thử xếp vào chiều cao nội dung hiện tại, nhưng không nhỏ hơn chiều cao của hình cao nhất
            target_repack_height = max(current_content_h, max_individual_rect_height)
            target_repack_height = max(target_repack_height, 1.0) # Chiều cao tối thiểu là 1 nếu có hình chữ nhật

        if target_repack_height == 0 and rects_to_repack_copies: # Bất thường
            target_repack_height = max(r.height for r in rects_to_repack_copies) # Fallback

        # 3. Thực hiện xếp lại
        # Tạo một shell lớp tạm thời với chiều cao mục tiêu mới
        # ID có thể giữ nguyên hoặc là một ID tạm thời, không quá quan trọng cho shell này
        temp_repacked_layer_shell = Layer(layer_id=layer_obj.id, # Giữ ID gốc để dễ theo dõi
                                        strip_width=layer_obj.strip_width,
                                        allocated_strip_height=target_repack_height)

        # Gọi packer để xếp các bản sao vào shell tạm thời
        # packer.run_packing_for_layer sẽ cập nhật temp_repacked_layer_shell.rectangles,
        # .content_actual_height, .total_profit, .cutting_commands
        self.packer.run_packing_for_layer(temp_repacked_layer_shell, rects_to_repack_copies)

        # 4. Cập nhật layer_obj gốc với kết quả xếp lại
        # Ngay cả khi số lượng hình chữ nhật xếp được ít hơn (do target_repack_height quá nhỏ),
        # chúng ta vẫn cập nhật, vì mục tiêu là giảm chiều cao.
        # Tuy nhiên, logic này giả định rằng tất cả các hình chữ nhật trong layer_obj.rectangles
        # PHẢI được xếp lại. Nếu không, đây không phải là "repacking" mà là "re-solving".
        # Giả định: packer sẽ cố gắng xếp tất cả các hình trong rects_to_repack_copies.
        # Nếu target_repack_height quá nhỏ, packer có thể không xếp được gì cả.

        # Một cách tiếp cận an toàn hơn là chỉ cập nhật nếu số lượng hình chữ nhật không giảm
        # và lợi nhuận không giảm (hoặc chỉ chấp nhận nếu chiều cao giảm đáng kể).
        # Bài báo chỉ nói "tries to reduce the height".

        # Cập nhật layer_obj với thông tin từ temp_repacked_layer_shell
        layer_obj.rectangles = temp_repacked_layer_shell.rectangles
        layer_obj.cutting_commands = temp_repacked_layer_shell.cutting_commands
        
        # Tính toán lại chiều cao nội dung và lợi nhuận dựa trên kết quả xếp lại mới
        new_content_height, new_profit = layer_obj.calculate_content_dims_and_profit()
        
        # Quan trọng: Cập nhật allocated_strip_height của lớp thành chiều cao nội dung mới
        # Đây chính là bước "reduce the height of a layer"
        layer_obj.allocated_strip_height = new_content_height
        
        # print(f"    Layer {layer_obj.id} repacked. New allocated H: {layer_obj.allocated_strip_height:.2f}, New content H: {new_content_height:.2f}, New profit: {new_profit:.2f}")


    def _profit_repair_local_search(self, layer_obj: Layer, candidate_rects_for_swap: list): #
        """
        Cố gắng cải thiện lợi nhuận của layer_obj bằng cách hoán đổi các hình chữ nhật
        từ candidate_rects_for_swap với các hình chữ nhật hiện có trong lớp.
        Chấp nhận hoán đổi tốt nhất cho mỗi ứng viên nếu lợi nhuận tăng
        mà không tăng chiều cao cấp phát ban đầu của lớp.
        Sửa đổi trực tiếp layer_obj.
        """
        if not candidate_rects_for_swap or not layer_obj.rectangles:
            # print(f"    ProfitRepair: No candidates or layer is empty. Layer {layer_obj.id}.")
            return

        # print(f"Framework: LS - ProfitRepair for layer {layer_obj.id} with {len(candidate_rects_for_swap)} candidates. Initial LProfit: {layer_obj.total_profit:.0f}, AllocH: {layer_obj.allocated_strip_height:.0f}")

        # Tạo bản sao danh sách ứng viên để có thể xóa mà không ảnh hưởng đến danh sách gốc truyền vào
        # Giả định candidate_rects_for_swap là list các Rectangle objects
        list_of_candidate_rects = [deepcopy(r) for r in candidate_rects_for_swap]
        
        # Theo dõi xem có thay đổi nào được thực hiện trong toàn bộ quá trình không
        overall_change_made = False

        # Lặp qua từng ứng viên
        # Sử dụng chỉ số để có thể xóa ứng viên đã được dùng thành công
        candidate_idx = 0
        while candidate_idx < len(list_of_candidate_rects):
            r_candidate = list_of_candidate_rects[candidate_idx]
            
            if not layer_obj.rectangles: # Nếu lớp trở nên rỗng
                break

            # Với mỗi r_candidate, tìm hoán đổi tốt nhất có thể với một hình trong lớp hiện tại
            best_swap_info = {
                "r_in_layer_to_remove": None, # Hình chữ nhật trong lớp sẽ bị loại bỏ
                "temp_layer_after_swap": None, # Trạng thái lớp tạm thời sau khi hoán đổi và repacking
                "profit_gain": -float('inf') # Mức tăng lợi nhuận
            }
            
            # Trạng thái lớp TRƯỚC KHI thử bất kỳ hoán đổi nào cho r_candidate này
            # Chúng ta so sánh mọi hoán đổi tiềm năng với trạng thái này của lớp.
            current_layer_profit_before_this_candidate = layer_obj.total_profit
            current_layer_allocated_H_before_this_candidate = layer_obj.allocated_strip_height

            for r_in_layer_original_idx in range(len(layer_obj.rectangles)):
                r_in_layer_to_swap_out = layer_obj.rectangles[r_in_layer_original_idx]

                # 1. Tạo danh sách hình chữ nhật mới cho lớp nếu hoán đổi xảy ra
                temp_rect_list_for_swap_attempt = [
                    deepcopy(r) for i, r in enumerate(layer_obj.rectangles) if i != r_in_layer_original_idx
                ]
                temp_rect_list_for_swap_attempt.append(deepcopy(r_candidate))

                # 2. Tạo shell lớp tạm thời để thử xếp lại
                # Xếp lại vào chiều cao cấp phát HIỆN TẠI của lớp (trước khi r_candidate này được thử)
                # Điều này đảm bảo so sánh "không làm tăng chiều cao lớp" là công bằng.
                temp_swapped_layer_shell = Layer(
                    layer_id=layer_obj.id,
                    strip_width=self.W_strip,
                    allocated_strip_height=current_layer_allocated_H_before_this_candidate
                )
                
                for r_copy in temp_rect_list_for_swap_attempt:
                    r_copy.is_placed = False; r_copy.x=None; r_copy.y=None

                self.packer.run_packing_for_layer(temp_swapped_layer_shell, temp_rect_list_for_swap_attempt)
                self._repacking_local_search(temp_swapped_layer_shell) # Repacking được áp dụng

                # 3. Kiểm tra điều kiện chấp nhận
                profit_after_swap = temp_swapped_layer_shell.total_profit
                height_after_swap_and_repack = temp_swapped_layer_shell.allocated_strip_height # Đây là content_height mới

                if profit_after_swap > current_layer_profit_before_this_candidate and \
                height_after_swap_and_repack <= current_layer_allocated_H_before_this_candidate:
                    
                    current_profit_gain = profit_after_swap - current_layer_profit_before_this_candidate
                    if current_profit_gain > best_swap_info["profit_gain"]:
                        best_swap_info["profit_gain"] = current_profit_gain
                        best_swap_info["r_in_layer_to_remove"] = r_in_layer_to_swap_out # Lưu đối tượng gốc
                        best_swap_info["temp_layer_after_swap"] = temp_swapped_layer_shell
            
            # Kết thúc vòng lặp tìm hoán đổi tốt nhất cho r_candidate hiện tại

            if best_swap_info["temp_layer_after_swap"] is not None:
                # Thực hiện hoán đổi tốt nhất đã tìm thấy
                # print(f"    ProfitRepair: ACCEPTED swap for L{layer_obj.id}. Candidate R{r_candidate.id} swapped with L_R{best_swap_info['r_in_layer_to_remove'].id}. "
                #       f"Profit: {current_layer_profit_before_this_candidate:.0f} -> {best_swap_info['temp_layer_after_swap'].total_profit:.0f}. "
                #       f"Alloc H: {current_layer_allocated_H_before_this_candidate:.0f} -> {best_swap_info['temp_layer_after_swap'].allocated_strip_height:.0f}")

                final_swapped_layer_state = best_swap_info["temp_layer_after_swap"]
                layer_obj.rectangles = final_swapped_layer_state.rectangles
                layer_obj.cutting_commands = final_swapped_layer_state.cutting_commands
                layer_obj.allocated_strip_height = final_swapped_layer_state.allocated_strip_height
                layer_obj.calculate_content_dims_and_profit() # Cập nhật profit, content_actual_height

                # r_candidate đã được sử dụng, loại bỏ nó khỏi danh sách ứng viên
                list_of_candidate_rects.pop(candidate_idx)
                
                # Hình chữ nhật r_in_layer_to_remove đã bị loại bỏ khỏi lớp.
                # Nó nên được trả lại cho rem_set của individual.
                # Hàm này chỉ sửa layer_obj. Hàm gọi nó (_sample_ECM_for_layers_and_pack)
                # cần cập nhật rem_set của individual bằng cách thêm best_swap_info['r_in_layer_to_remove']
                # và xóa r_candidate khỏi danh sách các hình chưa được chọn của QK.
                # Để đơn giản, chúng ta chỉ đánh dấu có thay đổi.
                overall_change_made = True
                
                # Không tăng candidate_idx vì list_of_candidate_rects đã bị rút ngắn
                # Vòng lặp while sẽ tự xử lý việc này.
                # Chúng ta đã xử lý xong r_candidate này.
            else:
                candidate_idx += 1 # Chuyển sang ứng viên tiếp theo nếu không có swap nào tốt cho ứng viên này

        # if overall_change_made:
            # print(f"    ProfitRepair for layer {layer_obj.id} made changes. Final LProfit: {layer_obj.total_profit:.0f}, AllocH: {layer_obj.allocated_strip_height:.0f}")
        # else:
            # print(f"    ProfitRepair for layer {layer_obj.id} made NO changes.")

    def _repair_procedure(self, individual: SolutionIndividual): #
        """
        Nếu tổng chiều cao các lớp của individual vượt quá H_strip,
        lặp đi lặp lại việc xóa một hình chữ nhật ngẫu nhiên từ một lớp ngẫu nhiên,
        thêm nó vào rem_set, và áp dụng repacking cho lớp bị sửa đổi,
        cho đến khi individual trở nên khả thi hoặc không còn gì để xóa.
        Sửa đổi trực tiếp individual.
        """
        # print(f"Framework: LS - RepairProcedure for individual. Initial total H: {individual.total_allocated_height:.2f} / {self.H_strip:.2f}")

        # Tính toán lại tổng chiều cao để chắc chắn
        individual.calculate_fitness_and_total_height()

        # "If the total height of the layers is too large (it is larger as H)"
        while not individual.is_feasible(): # individual.total_allocated_height > self.H_strip
            # Tìm tất cả các lớp có chứa hình chữ nhật để có thể xóa
            eligible_layers_for_deletion = [layer for layer in individual.layers if layer.rectangles]

            if not eligible_layers_for_deletion:
                # Không còn hình chữ nhật nào trong bất kỳ lớp nào để xóa
                # print("    RepairProcedure: No more rectangles to delete, but individual still infeasible.")
                break 

            # "deletes a random rectangle of a random layer"
            random_layer_to_modify = random.choice(eligible_layers_for_deletion)
            
            if not random_layer_to_modify.rectangles : # Kiểm tra lại, dù đã lọc ở trên
                continue

            rect_to_delete = random.choice(random_layer_to_modify.rectangles)
            
            # Xóa hình chữ nhật khỏi lớp
            random_layer_to_modify.rectangles.remove(rect_to_delete) # Xóa theo đối tượng
            
            # "The deleted rectangles will store in the rem set."
            individual.rem_set.append(deepcopy(rect_to_delete)) # Thêm bản sao vào rem_set

            # "applies the Repacking local search."
            self._repacking_local_search(random_layer_to_modify)
            
            # Cập nhật lại tổng chiều cao và fitness của individual
            individual.calculate_fitness_and_total_height()

            # "If the new total height is smaller as H, the improving is ready."
            # Vòng lặp while sẽ tự kiểm tra điều kiện này.
            # "Otherwise the procedure repeats the process; deleting another rectangle."
            # print(f"    RepairProcedure: Deleted R{rect_to_delete.id} from L{random_layer_to_modify.id}. New total H: {individual.total_allocated_height:.2f}")


        # print(f"Framework: LS - RepairProcedure finished. Final total H: {individual.total_allocated_height:.2f}. Feasible: {individual.is_feasible()}")

    def _LSrem_generic(self, individual: SolutionIndividual, num_rects_to_insert_type: str): #
        """
        Thực hiện logic chung cho LSrem1, LSrem2, LSrem3.
        num_rects_to_insert_type: 'single', 'group', 'all'.
        Sửa đổi trực tiếp 'individual' nếu tìm thấy cải tiến.
        Sử dụng chiến lược "first accept".
        """
        if not individual.rem_set or not individual.layers:
            # print(f"    LSrem ({num_rects_to_insert_type}): No items in rem_set or no layers in individual. Skipping.")
            return False # Không có thay đổi nào được thực hiện

        rects_to_try_insert = []
        made_change_to_individual = False # Cờ để theo dõi nếu có thay đổi

        # Bước 1: Chọn hình chữ nhật từ rem_set
        if num_rects_to_insert_type == 'single': # LSrem1 [cite: 221]
            if individual.rem_set:
                rects_to_try_insert = [random.choice(individual.rem_set)]
        elif num_rects_to_insert_type == 'group': # LSrem2
            if len(individual.rem_set) >= 1: # Cần ít nhất 1 để tạo nhóm
                # Kích thước nhóm ngẫu nhiên, ví dụ từ 1 đến min(kích thước cố định, số lượng còn lại)
                max_group_size = min(5, len(individual.rem_set)) # Ví dụ: nhóm tối đa 5
                if max_group_size > 0:
                    group_size = random.randint(1, max_group_size)
                    rects_to_try_insert = random.sample(individual.rem_set, group_size)
        elif num_rects_to_insert_type == 'all': # LSrem3
            rects_to_try_insert = individual.rem_set[:] # Lấy bản sao của tất cả
        else:
            # print(f"    LSrem: Unknown num_rects_to_insert_type: {num_rects_to_insert_type}")
            return False

        if not rects_to_try_insert:
            # print(f"    LSrem ({num_rects_to_insert_type}): No rectangles selected from rem_set to try inserting.")
            return False

        original_individual_fitness = individual.fitness # Lưu fitness ban đầu

        # Bước 2: Lặp qua các lớp đích để thử chèn
        for target_layer_idx in range(len(individual.layers)):
            target_layer_original_ref = individual.layers[target_layer_idx]

            # Bước 3: Mô phỏng chèn và xếp lại
            # Tạo danh sách các hình chữ nhật mới cho lớp = hình cũ của lớp + hình muốn chèn
            # Sử dụng deepcopy cho các hình chữ nhật để thử nghiệm không ảnh hưởng đến trạng thái gốc nếu thất bại
            current_layer_rects_copies = [deepcopy(r) for r in target_layer_original_ref.rectangles]
            rects_to_insert_copies = [deepcopy(r) for r in rects_to_try_insert]
            
            potential_rects_for_layer = current_layer_rects_copies + rects_to_insert_copies
            
            # Tính chiều cao của các lớp khác trong individual
            height_of_other_layers = sum(
                l.allocated_strip_height for i, l in enumerate(individual.layers) if i != target_layer_idx
            )

            # "give the possible highest height to the layer"
            max_possible_height_for_modified_layer = self.H_strip - height_of_other_layers

            # Kiểm tra xem chiều cao có đủ cho ít nhất một hình không (tối thiểu là hình cao nhất trong potential_rects_for_layer)
            min_h_needed_for_potential = 0
            if potential_rects_for_layer:
                min_h_needed_for_potential = max(r.height for r in potential_rects_for_layer) # Chiều cao của hình cao nhất
            
            if max_possible_height_for_modified_layer < min_h_needed_for_potential and potential_rects_for_layer:
                # print(f"    LSrem ({num_rects_to_insert_type}) on L{target_layer_idx}: Max possible height {max_possible_height_for_modified_layer:.2f} too small. Skipping.")
                continue # Chuyển sang lớp đích tiếp theo

            # Tạo một shell lớp tạm thời để xếp lại
            temp_modified_layer = Layer(layer_id=target_layer_original_ref.id,
                                        strip_width=self.W_strip,
                                        allocated_strip_height=max_possible_height_for_modified_layer)
            
            # Reset is_placed cho tất cả các hình chữ nhật sẽ được xếp lại vào lớp tạm
            for r_copy in potential_rects_for_layer:
                r_copy.is_placed = False; r_copy.x=None; r_copy.y=None

            # Gọi packlayer
            self.packer.run_packing_for_layer(temp_modified_layer, potential_rects_for_layer)
            
            # Áp dụng Repacking procedure
            self._repacking_local_search(temp_modified_layer) # Sửa đổi temp_modified_layer trực tiếp

            # Bước 4: Kiểm tra và Chấp nhận Cải tiến
            # Tính fitness và tổng chiều cao mới nếu chấp nhận thay đổi này
            current_fitness_of_other_layers = sum(
                l.total_profit for i, l in enumerate(individual.layers) if i != target_layer_idx
            )
            # temp_modified_layer.total_profit đã được cập nhật bởi calculate_content_dims_and_profit bên trong repacking hoặc packer
            potential_new_total_fitness = current_fitness_of_other_layers + temp_modified_layer.total_profit
            
            # temp_modified_layer.allocated_strip_height cũng đã được tối ưu bởi _repacking_local_search
            potential_new_total_height = height_of_other_layers + temp_modified_layer.allocated_strip_height

            # Điều kiện chấp nhận từ bài báo [cite: 223, 225]
            if potential_new_total_fitness > original_individual_fitness and \
            potential_new_total_height <= self.H_strip:
                
                # print(f"    LSrem ({num_rects_to_insert_type}) on L{target_layer_idx}: ACCEPTED. "
                #       f"OldFit={original_individual_fitness:.2f}, NewFit={potential_new_total_fitness:.2f}. "
                #       f"OldTotalH={individual.total_allocated_height:.2f}, NewTotalH={potential_new_total_height:.2f}")

                # Chấp nhận thay đổi:
                # 1. Thay thế lớp cũ bằng lớp mới đã sửa đổi
                individual.layers[target_layer_idx] = temp_modified_layer
                
                # 2. Cập nhật rem_set: loại bỏ các hình chữ nhật đã được chèn thành công
                # (những hình từ rects_to_try_insert mà thực sự có mặt trong temp_modified_layer.rectangles)
                successfully_inserted_ids_this_attempt = {r.id for r in rects_to_try_insert if r.id in {pr.id for pr in temp_modified_layer.rectangles}}
                
                new_rem_set = []
                for r_rem_original in individual.rem_set:
                    if r_rem_original.id not in successfully_inserted_ids_this_attempt:
                        new_rem_set.append(r_rem_original) # Giữ lại các hình chưa được chèn
                individual.rem_set = new_rem_set
                
                # 3. Tính toán lại fitness và tổng chiều cao của toàn bộ individual
                individual.calculate_fitness_and_total_height()
                made_change_to_individual = True
                return True # Chấp nhận cải tiến đầu tiên và kết thúc hàm _LSrem_generic

        # Nếu không có thay đổi nào được chấp nhận sau khi thử tất cả các lớp
        if not made_change_to_individual:
            pass
            # print(f"    LSrem ({num_rects_to_insert_type}): No improvement found after trying all layers.")
        return False # Không có thay đổi nào được thực hiện


    def _apply_main_local_searches(self, individual: SolutionIndividual):
        """
        Áp dụng nhóm các tìm kiếm cục bộ LSrem1, LSrem2, LSrem3.
        Nhóm này được lặp lại LSremn_iterations lần.
        """
        # print(f"Framework: Applying main local searches (LSrem group) for {self.LSremn_iterations} iterations...")
        if self.LSremn_iterations <= 0:
            return

        for i in range(self.LSremn_iterations):
            # print(f"  LSrem Iteration {i+1}/{self.LSremn_iterations}")
            
            # Thứ tự như trong bài báo: LSrem1 + LSrem2 + LSrem3 [cite: 226]
            # Hàm _LSrem_generic trả về True nếu có thay đổi được thực hiện (và rem_set có thể đã thay đổi)
            
            # LSrem1
            if not individual.rem_set: break # Nếu rem_set rỗng, không cần tiếp tục
            self._LSrem_generic(individual, 'single')
            
            # LSrem2
            if not individual.rem_set: break 
            self._LSrem_generic(individual, 'group')
            
            # LSrem3
            if not individual.rem_set: break 
            self._LSrem_generic(individual, 'all')
            
            if not individual.rem_set: # Kiểm tra lại sau cả nhóm
                # print(f"    LSrem Iteration {i+1}: rem_set became empty. Stopping LSrem iterations.")
                break 
        
        # Đảm bảo fitness và chiều cao được cập nhật sau tất cả các tìm kiếm cục bộ
        individual.calculate_fitness_and_total_height()
        # print(f"Framework: Finished main local searches. Final fitness for individual: {individual.fitness:.2f}")


    # --- Quản lý quần thể ---
    def _reinsertion(self, descendant: SolutionIndividual): #
        """
        Đưa hậu duệ vào quần thể.
        Hiện tại: Thêm hậu duệ vào, sắp xếp, và giữ lại tmax cá thể tốt nhất.
        Điều này đảm bảo nếu quần thể < tmax, hậu duệ được thêm vào.
        Nếu quần thể = tmax, hậu duệ chỉ được giữ lại nếu nó đủ tốt.
        """
        # print(f"Framework: Reinserting descendant (fitness {descendant.fitness:.2f}). Current pop size: {len(self.population)}")

        self.population.append(descendant)
        self.population.sort(key=lambda ind: ind.fitness, reverse=True) # Sắp xếp theo fitness giảm dần

        # Giữ lại self.population_size_max_tmax cá thể tốt nhất
        if len(self.population) > self.population_size_max_tmax:
            self.population = self.population[:self.population_size_max_tmax]
        
        self.current_population_size = len(self.population) # Cập nhật kích thước hiện tại
        # print(f"    New pop size: {self.current_population_size}. Best fitness in pop: {self.population[0].fitness if self.population else 'N/A':.2f}")


    def _restart_procedure(self): # [cite: 214]
        """
        Nếu giải pháp tốt nhất không thay đổi trong self.gp_restart_no_improvement_gens thế hệ,
        xóa self.rp_restart_delete_ratio tỷ lệ các cá thể yếu nhất khỏi quần thể.
        Reset self._generations_since_last_improvement.
        """
        # Biến self._generations_since_last_improvement được cập nhật trong vòng lặp chính
        if self._generations_since_last_improvement >= self.gp_restart_no_improvement_gens:
            if not self.population: # Không có gì để restart nếu quần thể rỗng
                self._generations_since_last_improvement = 0
                self._best_fitness_in_restart_period = -1.0
                return

            # print(f"Framework: *** RESTART PROCEDURE TRIGGERED after {self._generations_since_last_improvement} gens w/o improvement ***")
            
            num_to_delete = int(self.rp_restart_delete_ratio * len(self.population))
            
            if num_to_delete > 0:
                # Quần thể đã được sắp xếp, xóa các cá thể yếu nhất (ở cuối danh sách)
                self.population = self.population[:-num_to_delete] 
            
            self.current_population_size = len(self.population)
            self._generations_since_last_improvement = 0 # Reset bộ đếm
            
            # Cập nhật lại _best_fitness_in_restart_period dựa trên quần thể mới (nếu còn)
            if self.population:
                self._best_fitness_in_restart_period = self.population[0].fitness 
            else:
                self._best_fitness_in_restart_period = -1.0 # Hoặc một giá trị khởi tạo phù hợp
            
            # print(f"    Restart complete. New pop size: {self.current_population_size}. New best fitness in restart period: {self._best_fitness_in_restart_period:.2f}")


    # --- Vòng lặp tiến hóa chính ---
    def run_evolution_loop(self): # Algorithm 2
        # "Input: the instance, the values of the parameters." [cite: 199] (đã nhận qua __init__)
        # "Initial block building" [cite: 199] (trong self._initialize_master_rectangles)
        # "Every value of the probability models MI, ECM is 0.5." [cite: 199] (trong __init__)
        
        self._generate_initial_population() # "Generate the initial population." [cite: 199]
        
        if self.population: # "Update the probability models MI, ECM." [cite: 200]
            self._update_M1_model()
            self._update_ECM_model()
        
        # "Repeat until running time > timeend" [cite: 204]
        for gen_num in range(self.generations_count_limit):
            print(f"\n--- Generation {gen_num + 1} ---")
            
            # "Do kn times" [cite: 201] - Bài báo nói "2DKEDA generates only one descendent in every generation" [cite: 188]
            # và "kn - the algorithm is controlled in every knth generation" [cite: 196]
            # Có vẻ như vòng lặp "Do kn times" ở Algorithm 2 là để tạo kn hậu duệ *trước khi* cập nhật model / restart.
            # Tuy nhiên, để đơn giản và nhất quán với[cite: 188], ta tạo 1 hậu duệ mỗi lần.
            # Việc cập nhật model và restart sẽ xảy ra sau vòng lặp "Do kn times" trong Algorithm 2,
            # tức là sau mỗi `self.kn_generations_loop` thế hệ trong mô hình này.

            # Generate descendant (chọn một trong hai cách)
            descendant = None
            if random.random() < self.p_sampling_vs_mutation: # [cite: 253]
                descendant = self._generate_descendant_by_sampling() # "Generate the rectangles for the knapsack by sampling M1. Generate the layers by sampling ECM." [cite: 201]
            else:
                descendant = self._generate_descendant_by_mutation() # Sử dụng selection và ECM-based mutation

            # "Apply local searches." [cite: 201] (áp dụng cho hậu duệ vừa tạo)
            if descendant:
                self._apply_main_local_searches(descendant)
            
            # "Reinsertion." [cite: 202]
            if descendant:
                self._reinsertion(descendant)

            # "If (t < tmax) then t=t+1" [cite: 202] (đã được xử lý ngầm trong _reinsertion)
            
            # Các bước này trong Algorithm 2 dường như xảy ra *sau* vòng lặp "Do kn times",
            # tức là sau mỗi self.kn_generations_loop thế hệ.
            if (gen_num + 1) % self.kn_generations_loop == 0:
                # "Apply local searches on the last descendent, reinsertion." [cite: 202] (Đã làm ở trên cho mỗi hậu duệ)
                # "Apply local searches on the best individual, reinsertion." [cite: 203]
                if self.population:
                    best_current_individual = self.population[0] # Giả sử quần thể đã được sắp xếp
                    # Áp dụng LS cho cá thể tốt nhất. Cần cẩn thận nếu nó thay đổi fitness và vị trí.
                    # self._apply_main_local_searches(best_current_individual)
                    # self.population.sort(key=lambda ind: ind.fitness, reverse=True) # Sắp xếp lại
                    pass # Tạm bỏ qua LS trên best để tránh vòng lặp phức tạp

                # "Update the M1, ECM models." [cite: 203]
                self._update_M1_model()
                self._update_ECM_model()
                
                # "Restart." [cite: 203]
                self._restart_procedure()

            # Cập nhật trạng thái cho việc restart
            if self.population:
                current_best_fitness_in_pop = self.population[0].fitness
                if current_best_fitness_in_pop > self._best_fitness_in_restart_period:
                    self._best_fitness_in_restart_period = current_best_fitness_in_pop
                    self._generations_since_last_improvement = 0
                else:
                    self._generations_since_last_improvement += 1
            
            # Cập nhật giải pháp tốt nhất tổng thể
            if self.population and (self.best_solution_ever is None or self.population[0].fitness > self.best_solution_ever.fitness):
                self.best_solution_ever = deepcopy(self.population[0])
                print(f"    New Best Overall Fitness: {self.best_solution_ever.fitness:.2f}")
            
            print(f"    End of Gen {gen_num+1}. Pop best: {self.population[0].fitness if self.population else -1:.2f}. Overall: {self.best_solution_ever.fitness if self.best_solution_ever else -1:.2f}. Pop size: {len(self.population)}")

        return self.best_solution_ever


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Dữ liệu hình chữ nhật ví dụ (id, type_id, width, height, profit)
    example_rectangles_data = [
        {'id': 0, 'type_id': 0, 'width': 20, 'height': 30, 'profit': 700},
        {'id': 1, 'type_id': 1, 'width': 30, 'height': 20, 'profit': 800},
        {'id': 2, 'type_id': 2, 'width': 10, 'height': 50, 'profit': 500},
        {'id': 3, 'type_id': 0, 'width': 20, 'height': 30, 'profit': 700}, # Another of type 0
        {'id': 4, 'type_id': 3, 'width': 40, 'height': 10, 'profit': 400},
        {'id': 5, 'type_id': 4, 'width': 25, 'height': 25, 'profit': 625},
        {'id': 6, 'type_id': 5, 'width': 15, 'height': 35, 'profit': 525},
    ]
    strip_main_W = 70
    strip_main_H = 50

    # Tham số từ bài báo (một số giá trị ví dụ hoặc cần điều chỉnh theo instance size)
    # Tham khảo Mục 6.1 Parameter selection
    eda_parameters = {
        'p_block_building_prob': 0.5,       # [cite: 276] (implicit)
        't_pop_initial': 5,                 # [cite: 275]
        'tmax_pop_max': 10,                 # [cite: 275] (giảm để chạy nhanh ví dụ)
        'kn_loop_control': 3,               # [cite: 275] (giảm để chạy nhanh ví dụ)
        'generations_limit': 15,            # Giới hạn thế hệ cho ví dụ
        'alpha_learning': 0.2,              # [cite: 155] (ví dụ)
        'p_samp_vs_mutate': 1.0,            # (luân phiên 0 và 1 trong bài báo)
        'tp_selection_ratio': 0.1,          # [cite: 275]
        'p_bf_heuristic_prob': 0.8,         # Xác suất dùng Best-Fit (HP1)
        'p_impLS_prob': 0.0,                # [cite: 278] (0 cho set3/set5)
        'imax_fit_list': 50,                # [cite: 107] (giảm cho ví dụ)
        'LSremn_iters': 2,                  # [cite: 282] (giảm cho ví dụ)
        'gp_restart_gens': 20,              # [cite: 275] (giảm cho ví dụ)
        'rp_restart_ratio': 0.7,            # [cite: 275]
    }
    # Điều chỉnh LSremn_iters dựa trên số lượng hình chữ nhật
    num_rects = len(example_rectangles_data)
    if num_rects < 100: eda_parameters['LSremn_iters'] = 2 # [cite: 282] (giảm mạnh cho ví dụ)
    else: eda_parameters['LSremn_iters'] = 1 # [cite: 283] (giảm mạnh cho ví dụ)


    print("Initializing 2DKEDA Algorithm Framework...")
    solver = TwoDKnapsackEDAAlgorithm(example_rectangles_data, strip_main_W, strip_main_H, eda_parameters)

    print("\nStarting Evolution Loop...")
    final_best_solution = solver.run_evolution_loop()

    print("\n--- Evolution Finished ---")
    if final_best_solution and final_best_solution.fitness > 0:
        print(f"Best Solution Fitness: {final_best_solution.fitness:.2f}")
        print(f"Total Allocated Height: {final_best_solution.total_allocated_height:.2f} / {strip_main_H:.2f}")
        print(f"Number of Layers: {len(final_best_solution.layers)}")
        for i, layer in enumerate(final_best_solution.layers):
            print(f"  Layer {i+1} (ID {layer.id}): Allocated H={layer.allocated_strip_height:.2f}, "
                  f"Content H={layer.content_actual_height:.2f}, Profit={layer.total_profit:.2f}, "
                  f"NumRects={len(layer.rectangles)}")
        print(f"Number of Rectangles in Rem Set: {len(final_best_solution.rem_set)}")
    else:
        print("No effective solution found or population did not initialize properly.")
# if __name__ == "__main__":
#     # ... (Định nghĩa lớp Rectangle ở trên) ...

#     # Tạo một vài đối tượng Rectangle để thử nghiệm
#     rect1 = Rectangle(id=1, width=10, height=5)
#     rect2 = Rectangle(id=2, width=8, height=4)
#     rect3 = Rectangle(id=3, width=12, height=6) # Lớn hơn vùng
#     rect4 = Rectangle(id=4, width=5, height=3)
#     rect4.is_placed = True # Hình chữ nhật này đã được đặt

#     # Khởi tạo PackLayerProcedureDetailed với tham số ví dụ
#     example_params = {'imax_fit_list': 2} # Giới hạn fit_list để dễ kiểm tra
#     packer_test_obj = PackLayerProcedure(example_params)

#     # Giả lập trạng thái _Q_map cho packer_test_obj
#     # Thông thường, _Q_map được thiết lập bởi run_packing_for_layer
#     packer_test_obj._Q_map = {
#         rect1.id: rect1,
#         rect2.id: rect2,
#         rect3.id: rect3,
#         rect4.id: rect4
#     }

#     # Gọi hàm _get_fit_list để kiểm tra
#     test_region_w, test_region_h = 10, 5
#     print(f"Kiểm tra _get_fit_list với vùng ({test_region_w}, {test_region_h}):")
    
#     # Trước khi gọi, đảm bảo trạng thái is_placed là đúng
#     # rect1, rect2, rect3 chưa đặt. rect4 đã đặt.
#     rect1.is_placed = False
#     rect2.is_placed = False
#     rect3.is_placed = False
    
#     resulting_fit_list = packer_test_obj._get_fit_list(test_region_w, test_region_h)

#     print(f"Số lượng hình chữ nhật phù hợp: {len(resulting_fit_list)}")
#     for r in resulting_fit_list:
#         print(r)

#     # Kết quả mong đợi: rect1 và rect2 (nếu imax_fit_list >= 2).
#     # rect3 quá lớn. rect4 đã được đặt.
#     # Nếu imax_fit_list = 2, thì chỉ 2 hình đầu tiên phù hợp sẽ được trả về.