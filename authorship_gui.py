from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from authorship_mvp import render_report, run_analysis


class AuthorshipMVPApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Authorship MVP — интерфейс эксперта")
        self.root.geometry("980x720")

        self.query_path: Path | None = None
        self.sample_path: Path | None = None
        self.last_report: str = ""

        self._build_ui()

    def _build_ui(self) -> None:
        wrapper = ttk.Frame(self.root, padding=12)
        wrapper.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            wrapper,
            text="Сравнение исследуемого текста и образца",
            font=("Segoe UI", 14, "bold"),
        )
        title.pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(
            wrapper,
            text=(
                "Шаг 1: загрузите два .txt файла\n"
                "Шаг 2: нажмите «Запустить анализ»\n"
                "Шаг 3: просмотрите отчет и при необходимости сохраните его в .md"
            ),
        ).pack(anchor=tk.W, pady=(0, 12))

        files_frame = ttk.LabelFrame(wrapper, text="Файлы")
        files_frame.pack(fill=tk.X, pady=(0, 10))

        self.query_label = ttk.Label(files_frame, text="Исследуемый текст: не выбран")
        self.query_label.grid(row=0, column=0, sticky=tk.W, padx=8, pady=8)
        ttk.Button(files_frame, text="Выбрать", command=self._select_query).grid(
            row=0, column=1, sticky=tk.E, padx=8, pady=8
        )

        self.sample_label = ttk.Label(files_frame, text="Образец: не выбран")
        self.sample_label.grid(row=1, column=0, sticky=tk.W, padx=8, pady=8)
        ttk.Button(files_frame, text="Выбрать", command=self._select_sample).grid(
            row=1, column=1, sticky=tk.E, padx=8, pady=8
        )

        files_frame.columnconfigure(0, weight=1)

        actions = ttk.Frame(wrapper)
        actions.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(actions, text="Запустить анализ", command=self._run_analysis).pack(side=tk.LEFT)
        ttk.Button(actions, text="Сохранить отчет", command=self._save_report).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Очистить", command=self._clear_report).pack(side=tk.LEFT, padx=(8, 0))

        report_frame = ttk.LabelFrame(wrapper, text="Аналитический отчет")
        report_frame.pack(fill=tk.BOTH, expand=True)

        self.report_text = tk.Text(report_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(report_frame, orient=tk.VERTICAL, command=self.report_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_text.configure(yscrollcommand=scrollbar.set)

    def _select_query(self) -> None:
        path = filedialog.askopenfilename(title="Выберите исследуемый текст", filetypes=[("Text files", "*.txt")])
        if path:
            self.query_path = Path(path)
            self.query_label.config(text=f"Исследуемый текст: {self.query_path}")

    def _select_sample(self) -> None:
        path = filedialog.askopenfilename(title="Выберите образец", filetypes=[("Text files", "*.txt")])
        if path:
            self.sample_path = Path(path)
            self.sample_label.config(text=f"Образец: {self.sample_path}")

    def _run_analysis(self) -> None:
        if not self.query_path or not self.sample_path:
            messagebox.showwarning("Недостаточно данных", "Нужно выбрать оба файла (.txt).")
            return

        try:
            query_text = self.query_path.read_text(encoding="utf-8")
            sample_text = self.sample_path.read_text(encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Ошибка чтения", f"Не удалось прочитать файлы:\n{exc}")
            return

        result = run_analysis(query_text, sample_text)
        self.last_report = render_report(result, self.query_path.name, self.sample_path.name)

        self.report_text.delete("1.0", tk.END)
        self.report_text.insert("1.0", self.last_report)

    def _save_report(self) -> None:
        if not self.last_report.strip():
            messagebox.showinfo("Нет отчета", "Сначала выполните анализ.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить отчет",
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt")],
        )
        if not path:
            return

        try:
            Path(path).write_text(self.last_report, encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить отчет:\n{exc}")
            return

        messagebox.showinfo("Готово", f"Отчет сохранен:\n{path}")

    def _clear_report(self) -> None:
        self.last_report = ""
        self.report_text.delete("1.0", tk.END)


def main() -> None:
    root = tk.Tk()
    app = AuthorshipMVPApp(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
