import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import fitz
from PIL import Image, ImageTk

from functions import (
    get_collection_count,
    extract_pages_from_pdf,
    chunk_pdf_pages,
    add_to_collection,
    query_collection,
    clear_collection,
)


class PDFViewer:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="both", expand=True)
        self.doc = None
        self.page_num = 0
        self.total_pages = 0
        self.photo = None

        nav = ttk.Frame(self.frame)
        nav.pack(fill="x")
        ttk.Button(nav, text="◀", command=self.prev_page, width=3).pack(side="left")
        self.page_label = ttk.Label(nav, text="No PDF")
        self.page_label.pack(side="left", padx=10)
        ttk.Button(nav, text="▶", command=self.next_page, width=3).pack(side="left")

        self.canvas = tk.Canvas(self.frame, bg="gray80", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render())

        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)

    def open(self, path, page):
        try:
            self.doc = fitz.open(path)
            self.total_pages = len(self.doc)
            self.page_num = max(1, min(page, self.total_pages))
            self.render()
        except Exception as e:
            self.page_label.configure(text=f"Error: {e}")
            self.doc = None

    def render(self):
        if not self.doc:
            return
        page = self.doc[self.page_num - 1]
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        cw = self.canvas.winfo_width() or 600
        ch = self.canvas.winfo_height() or 400
        scale = min(cw / pix.width, ch / pix.height, 1.5)
        new_w = int(pix.width * scale)
        new_h = int(pix.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=self.photo)
        self.page_label.configure(text=f"Page {self.page_num}/{self.total_pages}")

    def prev_page(self):
        if self.doc and self.page_num > 1:
            self.page_num -= 1
            self.render()

    def next_page(self):
        if self.doc and self.page_num < self.total_pages:
            self.page_num += 1
            self.render()


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Document Searcher")
        self.root.geometry("1300x750")
        self.root.minsize(1000, 600)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.build_query_tab()
        self.build_ingest_tab()

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.count_label = ttk.Label(status_frame, text="")
        self.count_label.pack(side="left")
        ttk.Button(status_frame, text="Clear DB", command=self.clear_db).pack(side="right")
        self.update_count()

    def build_query_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Search")

        paned = ttk.PanedWindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        top = ttk.Frame(left)
        top.pack(fill="x", pady=(5, 5), padx=5)
        ttk.Label(top, text="Query:").pack(side="left")
        self.query_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.query_var, width=50).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(top, text="Search", command=self.search).pack(side="left")

        opts = ttk.Frame(left)
        opts.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(opts, text="Results:").pack(side="left")
        self.n_results_var = tk.StringVar(value="3")
        ttk.Spinbox(opts, from_=1, to=10, textvariable=self.n_results_var, width=4).pack(side="left", padx=5)
        ttk.Label(opts, text="  (double-click a result to view PDF)").pack(side="left")

        self.results_text = tk.Text(left, wrap="word", font=("Courier", 10), cursor="hand2")
        self.results_text.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        self.results_text.tag_configure("clickable", foreground="#2563eb", underline=1)
        self.results_text.tag_configure("bg_even", background="#f0f4f8")
        self.results_text.tag_configure("bg_odd", background="#ffffff")
        self.results_text.bind("<Button-1>", self.on_result_click)
        self.results_text.bind("<Key>", lambda e: "break")

        scroll = ttk.Scrollbar(self.results_text, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.viewer = PDFViewer(right)

    def build_ingest_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Ingest PDF")

        form = ttk.Frame(frame)
        form.pack(fill="x", pady=20, padx=20)

        ttk.Label(form, text="PDF File:").grid(row=0, column=0, sticky="w", pady=5)
        self.pdf_path_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.pdf_path_var, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(form, text="Browse", command=self.browse_pdf).grid(row=0, column=2, pady=5)

        ttk.Label(form, text="Chunk Size:").grid(row=1, column=0, sticky="w", pady=5)
        self.chunk_var = tk.StringVar(value="500")
        ttk.Entry(form, textvariable=self.chunk_var, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(form, text="Overlap:").grid(row=2, column=0, sticky="w", pady=5)
        self.overlap_var = tk.StringVar(value="100")
        ttk.Entry(form, textvariable=self.overlap_var, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        ttk.Button(frame, text="Ingest PDF", command=self.ingest).pack(pady=10)

        self.ingest_log = tk.Text(frame, wrap="word", state="disabled", height=8, font=("Courier", 10))
        self.ingest_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def log(self, widget, msg):
        widget.configure(state="normal")
        widget.insert("end", msg + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    def browse_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if path:
            self.pdf_path_var.set(path)

    def ingest(self):
        path = self.pdf_path_var.get()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Select a valid PDF file.")
            return

        self.log(self.ingest_log, f"Ingesting {os.path.basename(path)}...")

        def task():
            try:
                chunk_size = int(self.chunk_var.get())
                overlap = int(self.overlap_var.get())
            except ValueError:
                self.log(self.ingest_log, "Invalid chunk size or overlap.")
                return

            pages = extract_pages_from_pdf(path)
            if not pages:
                self.log(self.ingest_log, "No text extracted.")
                return

            basename = os.path.splitext(os.path.basename(path))[0]
            ids, documents, metadatas = chunk_pdf_pages(
                pages, chunk_size, overlap, basename, source_path=path
            )

            add_to_collection(documents, metadatas, ids)

            self.log(self.ingest_log, f"Added {len(ids)} chunks.")
            self.update_count()

        threading.Thread(target=task, daemon=True).start()

    def search(self):
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Enter a query.")
            return

        try:
            n = int(self.n_results_var.get())
        except ValueError:
            n = 3

        def task():
            try:
                results = query_collection(query, n_results=n)
            except Exception:
                self.root.after(0, lambda: self.show_results("Collection is empty.", []))
                return

            if results is None:
                self.root.after(0, lambda: self.show_results("Collection is empty.", []))
                return

            ids = results["ids"][0]
            docs = results["documents"][0]
            dists = results["distances"][0]
            metas = results["metadatas"][0]

            data = list(zip(ids, docs, dists, metas))
            self.results_data = data

            self.root.after(0, lambda: self.show_results(query, data))

        threading.Thread(target=task, daemon=True).start()

    def show_results(self, query, data):
        self.results_text.delete("1.0", "end")

        if not data:
            self.results_text.insert("1.0", query)
            return

        header = f"Query: {query}\n{'='*60}\n\n"
        self.results_text.insert("1.0", header)

        for i, (did, doc, dist, meta) in enumerate(data):
            tag = f"result_{i}"
            bg = "bg_even" if i % 2 == 0 else "bg_odd"

            conf = 1 - dist
            page = meta.get("page", "?")
            source = meta.get("source", "?")
            snippet = doc[:500] + ("..." if len(doc) > 500 else "")

            clickable_text = f"[{i+1}] Score: {conf:.4f}  Page: {page}  File: {os.path.basename(source)}"
            body = f"\n    {snippet}\n\n"

            self.results_text.insert("end", clickable_text, (tag, "clickable", bg))
            self.results_text.insert("end", body, (bg,))

    def on_result_click(self, event):
        idx = self.results_text.index(f"@{event.x},{event.y}")
        if not hasattr(self, "results_data") or not self.results_data:
            return "break"

        tags = self.results_text.tag_names(idx)
        for tag in tags:
            if tag.startswith("result_"):
                result_idx = int(tag.split("_")[1])
                _, _, _, meta = self.results_data[result_idx]
                source = meta.get("source", "")
                page = meta.get("page", 1)
                if source and os.path.isfile(source):
                    self.viewer.open(source, page)
                return "break"

        return "break"

    def clear_db(self):
        if not messagebox.askyesno("Confirm", "Delete the entire collection?"):
            return
        try:
            clear_collection()
            self.results_data = []
            self.show_results("", [])
            self.log(self.ingest_log, "Collection cleared.")
            self.update_count()
        except Exception:
            messagebox.showerror("Error", "No collection to delete.")

    def update_count(self):
        count = get_collection_count()
        self.count_label.configure(text=f"Documents in DB: {count}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
