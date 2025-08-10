# ui/chapters_tab.py
# -*- coding: utf-8 -*-
import os
import customtkinter as ctk
from tkinter import messagebox
from ui.context_menu import TextWidgetContextMenu
from utils import read_file, save_string_to_txt, clear_file_content

def detect_project_root():
    """
    自动检测项目根目录
    通过查找特定的标识文件来确定项目根目录
    """
    current_dir = os.getcwd()
    
    # 项目标识文件列表
    project_markers = [
        'main.py',
        'requirements.txt', 
        'README.md',
        'novel_generator',  # 文件夹
        'ui',               # 文件夹
        'config_manager.py',
        'llm_adapters.py'
    ]
    
    # 检查当前目录
    def check_directory(path):
        """检查目录是否包含项目标识文件"""
        score = 0
        for marker in project_markers:
            marker_path = os.path.join(path, marker)
            if os.path.exists(marker_path):
                score += 1
        return score
    
    # 检查当前目录及其父目录
    max_score = 0
    best_path = None
    search_path = current_dir
    
    # 最多向上查找5级目录
    for _ in range(5):
        score = check_directory(search_path)
        if score > max_score:
            max_score = score
            best_path = search_path
        
        # 如果找到足够多的标识文件，认为找到了项目根目录
        if score >= 3:
            break
            
        parent = os.path.dirname(search_path)
        if parent == search_path:  # 已到达根目录
            break
        search_path = parent
    
    # 如果找到的路径包含chapters文件夹，则认为是正确的项目根目录
    if best_path and os.path.exists(os.path.join(best_path, 'chapters')):
        return best_path
    
    # 如果当前目录就有chapters文件夹，直接返回当前目录
    if os.path.exists(os.path.join(current_dir, 'chapters')):
        return current_dir
        
    return best_path if max_score >= 2 else None

def build_chapters_tab(self):
    self.chapters_view_tab = self.tabview.add("Chapters Manage")
    self.chapters_view_tab.rowconfigure(0, weight=0)
    self.chapters_view_tab.rowconfigure(1, weight=1)
    self.chapters_view_tab.columnconfigure(0, weight=1)

    top_frame = ctk.CTkFrame(self.chapters_view_tab)
    top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    top_frame.columnconfigure(0, weight=0)
    top_frame.columnconfigure(1, weight=0)
    top_frame.columnconfigure(2, weight=0)
    top_frame.columnconfigure(3, weight=0)
    top_frame.columnconfigure(4, weight=1)

    prev_btn = ctk.CTkButton(top_frame, text="<< 上一章", command=self.prev_chapter, font=("Microsoft YaHei", 12))
    prev_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")

    next_btn = ctk.CTkButton(top_frame, text="下一章 >>", command=self.next_chapter, font=("Microsoft YaHei", 12))
    next_btn.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    self.chapter_select_var = ctk.StringVar(value="")
    self.chapter_select_menu = ctk.CTkOptionMenu(top_frame, values=[], variable=self.chapter_select_var, command=self.on_chapter_selected, font=("Microsoft YaHei", 12))
    self.chapter_select_menu.grid(row=0, column=2, padx=5, pady=5, sticky="w")

    save_btn = ctk.CTkButton(top_frame, text="保存修改", command=self.save_current_chapter, font=("Microsoft YaHei", 12))
    save_btn.grid(row=0, column=3, padx=5, pady=5, sticky="w")

    refresh_btn = ctk.CTkButton(top_frame, text="刷新章节列表", command=self.refresh_chapters_list, font=("Microsoft YaHei", 12))
    refresh_btn.grid(row=0, column=5, padx=5, pady=5, sticky="e")

    self.chapters_word_count_label = ctk.CTkLabel(top_frame, text="字数：0", font=("Microsoft YaHei", 12))
    self.chapters_word_count_label.grid(row=0, column=4, padx=(0,10), sticky="e")

    self.chapter_view_text = ctk.CTkTextbox(self.chapters_view_tab, wrap="word", font=("Microsoft YaHei", 12))
    
    def update_word_count(event=None):
        text = self.chapter_view_text.get("0.0", "end-1c")
        text_length = len(text)
        self.chapters_word_count_label.configure(text=f"字数：{text_length}")
    
    self.chapter_view_text.bind("<KeyRelease>", update_word_count)
    self.chapter_view_text.bind("<ButtonRelease>", update_word_count)
    TextWidgetContextMenu(self.chapter_view_text)
    self.chapter_view_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5, columnspan=6)

    self.chapters_list = []
    refresh_chapters_list(self)

def refresh_chapters_list(self):
    filepath = self.filepath_var.get().strip()
    
    # 如果路径为空，尝试自动检测项目根目录
    if not filepath:
        current_dir = os.getcwd()
        project_root = detect_project_root()
        if project_root:
            self.safe_log(f"⚠️ 未设置保存路径，已自动检测到项目根目录: {project_root}")
            self.safe_log("💡 建议在 '小说参数配置' 标签页中设置正确的保存路径")
            self.filepath_var.set(project_root)
            filepath = project_root
        else:
            self.safe_log("❌ 未设置保存路径且无法自动检测项目目录")
            self.safe_log("📋 请在 '小说参数配置' 标签页中设置保存路径")
            self.safe_log(f"💡 当前工作目录: {current_dir}")
            self.chapter_select_menu.configure(values=[])
            return
    
    chapters_dir = os.path.join(filepath, "chapters")
    
    # 如果chapters文件夹不存在，尝试创建它
    if not os.path.exists(chapters_dir):
        try:
            os.makedirs(chapters_dir, exist_ok=True)
            self.safe_log(f"📁 已创建 chapters 文件夹: {chapters_dir}")
        except Exception as e:
            self.safe_log(f"❌ 无法创建 chapters 文件夹: {e}")
            self.safe_log("📋 请检查保存路径是否正确，或手动创建 chapters 文件夹")
            self.chapter_select_menu.configure(values=[])
            return

    all_files = os.listdir(chapters_dir)
    chapter_nums = []
    for f in all_files:
        if f.startswith("chapter_") and f.endswith(".txt"):
            number_part = f.replace("chapter_", "").replace(".txt", "")
            if number_part.isdigit():
                chapter_nums.append(number_part)
    chapter_nums.sort(key=lambda x: int(x))
    self.chapters_list = chapter_nums
    self.chapter_select_menu.configure(values=self.chapters_list)
    current_selected = self.chapter_select_var.get()
    if current_selected not in self.chapters_list:
        if self.chapters_list:
            self.chapter_select_var.set(self.chapters_list[0])
            load_chapter_content(self, self.chapters_list[0])
        else:
            self.chapter_select_var.set("")
            self.chapter_view_text.delete("0.0", "end")

def on_chapter_selected(self, value):
    load_chapter_content(self, value)

def load_chapter_content(self, chapter_number_str):
    if not chapter_number_str:
        return
    filepath = self.filepath_var.get().strip()
    chapter_file = os.path.join(filepath, "chapters", f"chapter_{chapter_number_str}.txt")
    if not os.path.exists(chapter_file):
        self.safe_log(f"章节文件 {chapter_file} 不存在！")
        return
    content = read_file(chapter_file)
    self.chapter_view_text.delete("0.0", "end")
    self.chapter_view_text.insert("0.0", content)

def save_current_chapter(self):
    chapter_number_str = self.chapter_select_var.get()
    if not chapter_number_str:
        messagebox.showwarning("警告", "尚未选择章节，无法保存。")
        return
    filepath = self.filepath_var.get().strip()
    if not filepath:
        messagebox.showwarning("警告", "请先配置保存文件路径")
        return
    chapter_file = os.path.join(filepath, "chapters", f"chapter_{chapter_number_str}.txt")
    content = self.chapter_view_text.get("0.0", "end").strip()
    clear_file_content(chapter_file)
    save_string_to_txt(content, chapter_file)
    self.safe_log(f"已保存对第 {chapter_number_str} 章的修改。")

def prev_chapter(self):
    if not self.chapters_list:
        return
    current = self.chapter_select_var.get()
    if current not in self.chapters_list:
        return
    idx = self.chapters_list.index(current)
    if idx > 0:
        new_idx = idx - 1
        self.chapter_select_var.set(self.chapters_list[new_idx])
        load_chapter_content(self, self.chapters_list[new_idx])
    else:
        messagebox.showinfo("提示", "已经是第一章了。")

def next_chapter(self):
    if not self.chapters_list:
        return
    current = self.chapter_select_var.get()
    if current not in self.chapters_list:
        return
    idx = self.chapters_list.index(current)
    if idx < len(self.chapters_list) - 1:
        new_idx = idx + 1
        self.chapter_select_var.set(self.chapters_list[new_idx])
        load_chapter_content(self, self.chapters_list[new_idx])
    else:
        messagebox.showinfo("提示", "已经是最后一章了。")
