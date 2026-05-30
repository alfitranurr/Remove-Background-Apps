import gradio as gr
import torch
from PIL import Image, ImageOps
from transparent_background import Remover
import os
import tempfile

# Load background remover model
device = "cuda" if torch.cuda.is_available() else "cpu"
remover = Remover(device=device)

def process_batch(files):
    if not files:
        return None, []
    
    output_filepaths = []
    preview_images = []
    
    # Create a temporary directory to save high-quality PNG outputs
    temp_dir = tempfile.mkdtemp()
    
    for file_item in files:
        # Handle file paths robustly across different Gradio versions
        if isinstance(file_item, str):
            filepath = file_item
        elif hasattr(file_item, 'name'):
            filepath = file_item.name
        else:
            filepath = file_item.get('name', '')
            
        if not filepath:
            continue
            
        # Get original file name
        filename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Open image and apply EXIF orientation correction
        img = Image.open(filepath)
        img = ImageOps.exif_transpose(img)
        img_rgb = img.convert("RGB")
        
        # Run AI background remover (returns RGBA Image)
        result = remover.process(img_rgb, type='rgba')
        
        # Save output strictly as PNG with high quality
        out_filename = f"{name_without_ext}_nobg.png"
        out_path = os.path.join(temp_dir, out_filename)
        result.save(out_path, format="PNG")
        
        output_filepaths.append(out_path)
        preview_images.append(result)
        
    return output_filepaths, preview_images

# Custom CSS for Premium Dark UI Styling
custom_css = """
#col-container {
    max-width: 950px;
    margin: 0 auto;
    padding: 20px;
}
.header-title {
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-weight: 800;
}
.header-desc {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
    font-size: 1.1rem;
}
.footer-text {
    text-align: center;
    color: #64748b;
    margin-top: 30px;
    font-size: 0.9rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 20px;
}
.footer-text a {
    color: #818cf8;
    text-decoration: none;
}
.footer-text a:hover {
    text-decoration: underline;
}
"""

# Build Gradio Blocks Interface
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    css=custom_css,
    title="Batch AI Background Remover"
) as demo:
    
    with gr.Column(elem_id="col-container"):
        # Title and Description
        gr.HTML("""
        <h1 class="header-title">✂️ Batch AI Background Remover</h1>
        <p class="header-desc">
            Unggah banyak gambar sekaligus dan hapus latar belakangnya secara otomatis dengan hasil format PNG transparan asli.
        </p>
        """)
        
        # Hardware Status Banner
        if device == "cuda":
            gr.HTML(f"""
            <div style='text-align: center; background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); padding: 10px; border-radius: 12px; font-weight: bold; margin-bottom: 20px;'>
                ⚡ Akselerasi GPU Aktif ({torch.cuda.get_device_name(0)})
            </div>
            """)
        else:
            gr.HTML("""
            <div style='text-align: center; background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); padding: 10px; border-radius: 12px; font-weight: bold; margin-bottom: 20px;'>
                ⚠️ Berjalan di CPU (Pemrosesan batch mungkin memerlukan waktu beberapa detik per gambar)
            </div>
            """)
            
        # Upload & Output Section
        with gr.Row():
            with gr.Column():
                input_files = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="1. Unggah Gambar (Bisa seret & lepas banyak file)"
                )
                submit_btn = gr.Button("Mulai Hapus Background Massal", variant="primary")
            with gr.Column():
                output_files = gr.File(
                    label="2. Unduh Hasil (.png)",
                    file_count="multiple"
                )
                gallery = gr.Gallery(
                    label="3. Preview Hasil Transparan",
                    columns=3,
                    height="auto"
                )
        
        # Bind process function to button click
        submit_btn.click(
            fn=process_batch,
            inputs=[input_files],
            outputs=[output_files, gallery]
        )
        
        # Footer
        gr.HTML("""
        <div class="footer-text">
            Dikembangkan oleh <b>Al Fitra Nur Ramadhani</b> &bull; 
            <a href="https://github.com/alfitranurr/Remove-Background-Apps" target="_blank">Repositori GitHub</a><br>
            Powered by 🐍 Python, 🔥 PyTorch, & 🎨 Gradio
        </div>
        """)

# Launch
if __name__ == "__main__":
    demo.launch()
