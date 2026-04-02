import streamlit as st
import os

try:
    import streamlit.elements.image
    from streamlit.elements.lib.image_utils import image_to_url
    if not hasattr(streamlit.elements.image, 'image_to_url'):
        streamlit.elements.image.image_to_url = image_to_url
except ImportError:
    pass 

import yaml
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import time
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.inference import WoundAnalyzer
import plotly.express as px
from streamlit_image_comparison import image_comparison
from streamlit_drawable_canvas import st_canvas
import datetime

st.set_page_config(
    page_title="WoundSeg Pro | 创面智能分析平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* 引入 Inter 字体 (可选，或使用系统默认) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* 全局字体设置 */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        color: #1e293b; /* Slate 800 */
    }

    /* 标题样式优化 */
    h1 {
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #0f172a; /* Slate 900 */
        margin-bottom: 1.5rem;
    }
    h2, h3 {
        font-weight: 600;
        color: #334155; /* Slate 700 */
    }

    /* 侧边栏美化 */
    [data-testid="stSidebar"] {
        background-color: #f8fafc; /* Slate 50 */
        border-right: 1px solid #e2e8f0;
    }
    
    /* 卡片式容器样式 (辅助 st.container(border=True)) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        background-color: white !important;
        padding: 20px !important;
        transition: box-shadow 0.3s ease;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025) !important;
    }

    /* 指标组件美化 - 更醒目的设计 */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 16px !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }
    label[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        background: -webkit-linear-gradient(45deg, #2563eb, #1e40af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* 按钮样式升级 - 更加现代化 */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: 1px solid #e2e8f0 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    
    /* Primary Button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3) !important;
    }

    /* 侧边栏导航美化 - 类似菜单项 */
    div[data-testid="stSidebar"] div.stRadio > label {
        display: none !important; /* 隐藏标题 */
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 12px 16px !important;
        border-radius: 8px !important;
        margin-bottom: 4px !important;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #eff6ff !important;
        color: #2563eb !important;
    }
    /* 选中的项 */
    div[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #dbeafe !important;
        color: #1e40af !important;
        font-weight: 600 !important;
        border: 1px solid #bfdbfe !important;
    }

    /* 标题增强 */
    h1 {
        background: -webkit-linear-gradient(45deg, #0f172a, #334155);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 1rem;
    }

    /* 输入框微调 */
    .stTextInput input, .stNumberInput input {
        border-radius: 6px;
        border-color: #e2e8f0;
    }
    
    /* 去除顶部 padding */
    .block-container {
        padding-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

def add_history(image_name, pixel_area, actual_area):
    record = {
        "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image": image_name,
        "Pixel Area": pixel_area,
        "Actual Area (mm²)": actual_area if actual_area else "N/A"
    }
    st.session_state.history.insert(0, record) 

@st.cache_resource
def load_model_resources():
    config_path = "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], "best_model.pth")
    model_loaded = os.path.exists(checkpoint_path)
    
    analyzer = WoundAnalyzer(checkpoint_path=checkpoint_path, config=config)
    return analyzer, config, model_loaded

try:
    analyzer, config, model_loaded = load_model_resources()
except Exception as e:
    st.error(f"Failed to initialize model: {e}")
    st.stop()

st.sidebar.markdown("### 🔬 WoundSeg Pro")
st.sidebar.caption(f"v1.2.0 | Engine: {analyzer.device}")

if not model_loaded:
    st.sidebar.error("⚠️ 模型未加载 (best_model.pth)")

st.sidebar.markdown("---")
page = st.sidebar.radio("功能导航", ["首页概览", "智能诊断", "数据修正", "批量处理", "系统监控"], label_visibility="collapsed")

if page == "首页概览":
    st.title("创面智能分析平台")
    st.caption("基于 EfficientNet-B3 + U-Net 的高精度医学图像分割解决方案")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("#### 🔍 智能诊断")
            st.markdown("""
            上传单张图片，毫秒级生成高精度分割掩膜。
            *   **自动抗扰**: 智能填充反光空洞
            *   **实时校准**: 毫米级面积计算
            *   **可视化**: 交互式滑块对比
            """)
            if st.button("前往诊断 ->", key="go_inference"):
                st.info("请在左侧导航栏点击“智能诊断”")

    with col2:
        with st.container(border=True):
            st.markdown("#### 🛠️ 数据修正")
            st.markdown("""
            强大的交互式标注工具，让模型越用越准。
            *   **画笔工具**: 手动涂抹/擦除掩膜
            *   **边缘融合**: 自动平滑修正区域
            *   **闭环迭代**: 一键保存并重训模型
            """)
            
    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.markdown("#### 📂 批量处理")
            st.markdown("一次性处理海量实验数据，自动生成 CSV 统计报告。")
            
    with col4:
        with st.container(border=True):
            st.markdown("#### 📈 系统监控")
            st.markdown("实时监控模型训练状态、Loss 曲线及 Checkpoints 文件管理。")

elif page == "数据修正":
    st.title("数据集精修与迭代")
    
    images_dir = "data/processed/images"
    masks_dir = "data/processed/masks"
    
    has_synced_image = hasattr(st.session_state, 'sync_image') and st.session_state.sync_image is not None
    
    if not has_synced_image:
        st.info("💡 目前没有待修正的图片。\n\n请先前往 **智能诊断** 页面，上传或拍摄图片，分析完成后点击 **'🛠️ 同步到数据修正页面'**。")
        st.stop()

    selected_file = st.session_state.sync_filename
    image = st.session_state.sync_image
    mask = st.session_state.sync_mask
    mask = (mask > 127).astype(np.uint8)
    st.info(f"正在修正同步的图片: **{selected_file}**")

    if 'selected_file' in locals() and selected_file:
                st.markdown("### 🛠️ 交互修正工具")
                c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns(4)
                
                with c_ctrl1:
                    drawing_mode = st.radio("🖌️ 绘制模式", ("涂抹 (添加区域)", "擦除 (移除区域)"), horizontal=True, key="dataset_draw_mode")
                
                with c_ctrl2:
                    stroke_width = st.slider("📏 笔刷大小", 1, 50, 20, key="dataset_stroke_width")
                    
                with c_ctrl3:
                    morph_size = st.number_input(
                        "📐 边缘微调 (收缩/扩张)",
                        min_value=-20.0, max_value=20.0, value=0.0, step=0.5,
                        key="dataset_morph",
                        help="负值收缩，正值扩张"
                    )

                with c_ctrl4:
                    st.markdown("<br>", unsafe_allow_html=True) 
                    fill_holes = st.checkbox("✅ 自动填充孔洞", value=True, key="dataset_fill_holes", help="自动填补内部封闭的空洞")
                    
                    if st.button("🔄 重置掩膜", help="放弃当前所有未保存的修改"):
                        st.session_state.last_canvas_key = f"reset_{time.time()}"
                        st.rerun()

                st.markdown("---")

                col_canvas, col_preview = st.columns([1, 1])
                
                current_mask = mask.copy().astype(np.float32)

                if fill_holes:
                    mask_uint8 = (current_mask > 0.5).astype(np.uint8) * 255
                    
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    cv2.drawContours(mask_uint8, contours, -1, 255, thickness=cv2.FILLED)
                    
                    current_mask = (mask_uint8 > 127).astype(np.float32)
                
                if morph_size != 0:
                    scale = 2
                    current_mask_high = cv2.resize(current_mask, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    radius_high = int(abs(morph_size) * scale)
                    kernel_size = 2 * radius_high + 1
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    
                    if morph_size > 0:
                        current_mask_high = cv2.dilate(current_mask_high, kernel, iterations=1)
                    else:
                        current_mask_high = cv2.erode(current_mask_high, kernel, iterations=1)
                        
                    current_mask = cv2.resize(current_mask_high, (current_mask.shape[1], current_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

                with col_canvas:
                    st.markdown("##### 🎨 手动绘制区域")
                    
                    viz_bg = image.copy()
                    mask_overlay = (current_mask > 0.5).astype(np.uint8)
                    overlay_color = np.zeros_like(viz_bg)
                    overlay_color[:, :, 1] = 255 
                    
                    alpha = 0.4
                    viz_bg[mask_overlay==1] = cv2.addWeighted(viz_bg[mask_overlay==1], 1-alpha, overlay_color[mask_overlay==1], alpha, 0)
                    bg_pil = Image.fromarray(viz_bg)
                    
                    stroke_color = "#ffffff" if "涂抹" in drawing_mode else "#000000"
                    
                    
                    img_h, img_w = image.shape[:2]
                    canvas_width = 700 
                    if img_w < 700:
                        canvas_width = img_w
                    
                    ratio = img_h / img_w
                    canvas_height = int(canvas_width * ratio)
                    
                    canvas_key = f"canvas_dataset_edit_{selected_file}_{fill_holes}_{morph_size}"
                    
                    if 'last_canvas_key' not in st.session_state:
                        st.session_state.last_canvas_key = canvas_key
                    
                    if st.session_state.last_canvas_key != canvas_key:
                        st.toast("检测到参数变更，画布背景已更新（笔迹已重置）", icon="🔄")
                        st.session_state.last_canvas_key = canvas_key


                    bg_pil = bg_pil.resize((int(canvas_width), int(canvas_height)))

                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.0)",
                        stroke_width=stroke_width,
                        stroke_color=stroke_color,
                        background_image=bg_pil, 
                        update_streamlit=True,
                        height=int(canvas_height),
                        width=int(canvas_width),
                        drawing_mode="freedraw",
                        key=canvas_key,
                    )

                if canvas_result.image_data is not None:
                    stroke_data = canvas_result.image_data
                    stroke_data_resized = cv2.resize(stroke_data.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                    drawn_alpha = stroke_data_resized[:, :, 3]
                    drawn_mask = drawn_alpha > 1 
                    
                    if np.any(drawn_mask):
                        is_white = stroke_data_resized[:, :, 0] > 127
                        add_mask = drawn_mask & is_white
                        remove_mask = drawn_mask & (~is_white)
                        
                        if np.any(add_mask):
                            current_mask[add_mask] = 1.0
                        if np.any(remove_mask):
                            current_mask[remove_mask] = 0.0

                with col_preview:
                    st.markdown("##### 👁️ 最终效果预览")
                    
                    mask_viz = np.zeros_like(image)
                    mask_viz[:, :, 1] = current_mask * 255 
                    img_overlay = cv2.addWeighted(image, 1.0, mask_viz, 0.5, 0)
                    
                    image_comparison(
                        img1=image,
                        img2=img_overlay,
                        label1="原始图像",
                        label2="修正后掩膜",
                        width=700, 
                        starting_position=50,
                        show_labels=True,
                        make_responsive=True,
                        in_memory=True
                    )
                    
                    st.markdown("---")
                    
                    pixel_area = np.sum(current_mask)
                    pixels_per_mm = config['data'].get('pixels_per_mm', 42.0)
                    area_mm2 = pixel_area / (pixels_per_mm ** 2)
                    
                    m1, m2 = st.columns(2)
                    m1.metric("当前面积", f"{area_mm2:.2f} mm²")
                    
                    with m2:
                         if st.button("💾 保存并更新", type="primary", key="dataset_save_btn", use_container_width=True):
                            try:
                                save_mask = (current_mask * 255).astype(np.uint8)
                                
                                save_img_path = os.path.join(images_dir, selected_file)
                                save_mask_path = os.path.join(masks_dir, os.path.splitext(selected_file)[0] + '.png')
                                
                                os.makedirs(images_dir, exist_ok=True)
                                os.makedirs(masks_dir, exist_ok=True)
                                
                                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(save_img_path, img_bgr)
                                cv2.imwrite(save_mask_path, save_mask)
                                
                                st.session_state.sync_image = None
                                
                                st.success("✅ 已将新图片与修正后的掩膜成功添加到训练数据集中！")
                                time.sleep(1.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"保存失败: {e}")
                    
                    st.caption("提示: 点击保存后，当前效果将写入磁盘。如需重新训练，请点击下方按钮。")
                    
                    if st.button("🚀 启动模型重训", key="dataset_train_btn", use_container_width=True):
                        st.info("正在启动训练进程...")
                        import subprocess
                        import sys
                        try:
                            subprocess.Popen([sys.executable, "run.py", "train"], cwd=os.getcwd(), creationflags=subprocess.CREATE_NEW_CONSOLE)
                            st.toast("训练已后台启动！", icon="🚀")
                        except Exception as e:
                            st.error(f"启动失败: {e}")

elif page == "智能诊断":
    st.title("智能辅助诊断")
    
    col_input, col_result = st.columns([1, 2])
    
    with col_input:
        with st.container(border=True):
            st.markdown("#### 1. 图像上传或拍照")
            upload_mode = st.radio("选择图片获取方式", ["本地上传", "实时拍照"], horizontal=True, label_visibility="collapsed")
            if upload_mode == "本地上传":
                uploaded_file = st.file_uploader("支持 JPG, PNG, BMP", type=['jpg', 'jpeg', 'png', 'bmp'])
            else:
                uploaded_file = st.camera_input("使用摄像头拍照")
        
        if uploaded_file:
            with st.container(border=True):
                st.markdown("#### 2. 参数校准")
                with st.expander("⚙️ 高级设置", expanded=False):
                    threshold = st.slider("判定阈值 (Confidence)", 0.0, 1.0, 0.5, 0.05, help="模型判定为伤口的概率阈值")
                    
                    st.markdown("---")
                    st.markdown("**� 比例尺校准**")
                    manual_calibration = st.checkbox("启用手动校准", value=False)
                    
                    pixels_per_mm = config['data'].get('pixels_per_mm', 42.0)
                    if manual_calibration:
                        pixels_per_mm = st.number_input("像素/毫米 (px/mm)", value=pixels_per_mm, min_value=1.0)
                    else:
                        st.caption(f"默认比例: 1mm = {pixels_per_mm} px")
                    
                    st.markdown("---")
                    st.markdown("**🎨 形态学微调**")
                    morph_size = st.number_input(
                        "边缘收缩/扩张 (px)",
                        min_value=-20.0, max_value=20.0, value=0.0, step=0.5,
                        help="负值收缩，正值扩张"
                    )
                    
                    fill_holes = st.checkbox("✅ 自动填充孔洞", value=True)

    with col_result:
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
            tfile.write(uploaded_file.getvalue())
            tfile.close() 
            
            try:
                with st.spinner('🤖 AI 正在极速分析中，请稍候...'):
                    result = analyzer.analyze_image(tfile.name, threshold=threshold)
                    
                current_mask = result['mask'].copy()
                
                if fill_holes:
                    mask_uint8 = (current_mask > 0.5).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(mask_uint8, contours, -1, 255, thickness=cv2.FILLED)
                    current_mask = (mask_uint8 > 127).astype(np.float32)
                
                if morph_size != 0:
                    scale = 2
                    current_mask_high = cv2.resize(current_mask, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    radius_high = int(abs(morph_size) * scale)
                    kernel_size = 2 * radius_high + 1
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    if morph_size > 0:
                        current_mask_high = cv2.dilate(current_mask_high, kernel, iterations=1)
                    else:
                        current_mask_high = cv2.erode(current_mask_high, kernel, iterations=1)
                    current_mask = cv2.resize(current_mask_high, (current_mask.shape[1], current_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

                pixel_area = np.sum(current_mask)
                area_mm2 = pixel_area / (pixels_per_mm ** 2)
                
                
                m1, m2, m3 = st.columns(3)
                m1.metric("伤口面积", f"{area_mm2:.2f} mm²")
                m2.metric("像素统计", f"{int(pixel_area)} px")
                m3.metric("置信度阈值", f"{threshold:.2f}")
                
                st.markdown("---")
                
                img = result['original_image']
                mask_viz = np.zeros_like(img)
                mask_viz[:, :, 1] = current_mask * 255 
                
                overlay = cv2.addWeighted(img, 1.0, mask_viz, 0.4, 0)
                
                mask_uint8 = (current_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                
                st.markdown("#### 👁️ 视觉对比")
                image_comparison(
                    img1=img,
                    img2=overlay,
                    label1="原始图像",
                    label2="智能分割",
                    width=700,
                    starting_position=50,
                    show_labels=True,
                    make_responsive=True,
                    in_memory=True
                )

                st.markdown("---")
                st.markdown("#### 💾 结果导出")
                
                _, mask_buffer = cv2.imencode(".png", (current_mask * 255).astype(np.uint8))
                
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                _, overlay_buffer = cv2.imencode(".jpg", overlay_bgr)
                
                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        label="📥 下载掩膜 (PNG)",
                        data=mask_buffer.tobytes(),
                        file_name="mask.png",
                        mime="image/png"
                    )
                with d2:
                    st.download_button(
                        label="📥 下载叠加图 (JPG)",
                        data=overlay_buffer.tobytes(),
                        file_name="overlay.jpg",
                        mime="image/jpeg"
                    )
                
                if 'history' not in st.session_state:
                    st.session_state.history = []
                
                if st.button("📥 记录此结果到历史"):
                    record = {
                        "时间": time.strftime("%H:%M:%S"),
                        "文件名": uploaded_file.name,
                        "面积(mm²)": round(area_mm2, 2),
                        "像素(px)": int(pixel_area)
                    }
                    st.session_state.history.append(record)
                    st.toast("已记录到历史！", icon="📝")
                    
                if st.button("🛠️ 同步到数据修正页面"):
                    st.session_state.sync_image = img
                    st.session_state.sync_mask = (current_mask * 255).astype(np.uint8)
                    st.session_state.sync_filename = uploaded_file.name
                    st.toast("已同步！请点击左侧导航栏前往【数据修正】页面", icon="✅")
                
                if st.session_state.history:
                    with st.expander("📚 分析历史记录", expanded=True):
                        history_df = pd.DataFrame(st.session_state.history)
                        st.dataframe(history_df, use_container_width=True)
                        
                        csv_history = history_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 导出历史记录 (CSV)",
                            data=csv_history,
                            file_name=f"wound_history_{time.strftime('%Y%m%d_%H%M')}.csv",
                            mime='text/csv'
                        )

            except Exception as e:
                st.error(f"处理失败: {e}")
            finally:
                if os.path.exists(tfile.name):
                    try:
                        os.unlink(tfile.name)
                    except PermissionError:
                        pass
        else:
            st.info("👈 请在左侧上传图片以开始分析")

elif page == "批量处理":
    st.title("📂 批量样本分析")
    
    with st.container(border=True):
        st.markdown("#### 1. 数据导入")
        uploaded_files = st.file_uploader("选择要分析的图片（支持多选/框选）", accept_multiple_files=True, type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_files:
        st.markdown("#### 2. 处理队列")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"已选择 {len(uploaded_files)} 张图片")
        with col2:
            start_btn = st.button("▶️ 开始批量处理", type="primary", use_container_width=True)
            
        if start_btn:
            output_csv = "batch_results.csv"
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            total_files = len(uploaded_files)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                masks_out_dir = os.path.join(temp_dir, "masks")
                os.makedirs(masks_out_dir, exist_ok=True)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"正在处理 ({i+1}/{total_files}): {uploaded_file.name}")
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        res = analyzer.analyze_image(temp_path)
                        results.append({
                            'image_name': uploaded_file.name,
                            'pixel_area': res['pixel_area'],
                            'actual_area_mm2': res['actual_area_mm2']
                        })
                        
                        mask_filename = os.path.splitext(uploaded_file.name)[0] + "_mask.png"
                        cv2.imwrite(os.path.join(masks_out_dir, mask_filename), res['mask'] * 255)
                        
                    except Exception as e:
                        st.warning(f"跳过 {uploaded_file.name}: {e}")
                    
                    progress_bar.progress((i + 1) / total_files)
                
                if results:
                    df = pd.DataFrame(results)
                    df.to_csv(os.path.join(masks_out_dir, "batch_results.csv"), index=False)
                    
                    shutil.make_archive(os.path.join(temp_dir, "wound_analysis_results"), 'zip', masks_out_dir)
                    with open(os.path.join(temp_dir, "wound_analysis_results.zip"), "rb") as f:
                        zip_bytes = f.read()
                else:
                    zip_bytes = None

            status_text.success("✅ 批量处理完成！")
            
            if results:
                st.markdown("#### 3. 结果报告")
                st.dataframe(df, use_container_width=True)
                
                if zip_bytes:
                    st.download_button(
                        label="📦 一键下载完整数据包 (包含所有掩膜与 CSV 报告)",
                        data=zip_bytes,
                        file_name=f'wound_analysis_results_{time.strftime("%Y%m%d_%H%M")}.zip',
                        mime='application/zip',
                        type="primary",
                        use_container_width=True
                    )

            else:
                st.warning("未生成任何结果。")

elif page == "系统监控":
    st.title("📈 系统与训练监控")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("#### 📊 训练曲线")
            log_file = os.path.join(config['paths'].get('log_dir', 'logs'), 'training_log.csv')
            
            if os.path.exists(log_file):
                try:
                    df_log = pd.read_csv(log_file)
                    if not df_log.empty:
                        fig_loss = px.line(df_log, x='epoch', y=['train_loss', 'val_loss'], 
                                         title='Loss Curve', labels={'value': 'Loss', 'variable': 'Type'})
                        st.plotly_chart(fig_loss, use_container_width=True)
                        
                        fig_dice = px.line(df_log, x='epoch', y=['train_dice', 'val_dice'], 
                                         title='Dice Score Curve', labels={'value': 'Dice Score', 'variable': 'Type'})
                        st.plotly_chart(fig_dice, use_container_width=True)
                    else:
                        st.info("日志文件为空，等待训练数据...")
                except Exception as e:
                    st.error(f"读取日志失败: {e}")
            else:
                st.info("暂无训练日志。请运行训练以生成数据。")

    with col2:
        with st.container(border=True):
            st.markdown("#### ⚙️ 训练配置")
            st.json(config['training'])
            
        with st.container(border=True):
            st.markdown("#### 💾 模型检查点")
            checkpoint_dir = config['paths']['checkpoint_dir']
            if os.path.exists(checkpoint_dir):
                files = os.listdir(checkpoint_dir)
                if files:
                    st.success(f"找到 {len(files)} 个检查点文件")
                    st.code("\n".join(files))
                else:
                    st.warning("目录为空")
            else:
                st.error("Checkpoints 目录不存在")
