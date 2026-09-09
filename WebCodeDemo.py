import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from catboost import CatBoostClassifier
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# =========================================================
# 1. CẤU HÌNH TRANG WEB
# =========================================================

st.set_page_config(
    page_title="Phân tích rủi ro bảo hiểm xe cơ giới",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. ĐƯỜNG DẪN ĐẾN CÁC MÔ HÌNH
# =========================================================

THU_MUC_MO_HINH = "models"

DUONG_DAN_MO_HINH_RUI_RO = os.path.join(
    THU_MUC_MO_HINH,
    "catboost_risk.cbm"
)

DUONG_DAN_MO_HINH_DINH_GIA = os.path.join(
    THU_MUC_MO_HINH,
    "tweedie_pricing.pkl"
)

DUONG_DAN_BO_HIEU_CHINH = os.path.join(
    THU_MUC_MO_HINH,
    "probability_calibrator.pkl"
)

DUONG_DAN_NGUONG_RUI_RO = os.path.join(
    THU_MUC_MO_HINH,
    "risk_threshold.pkl"
)

DUONG_DAN_TIEN_XU_LY = os.path.join(
    THU_MUC_MO_HINH,
    "preprocessor.pkl"
)

DUONG_DAN_CAU_HINH = os.path.join(
    THU_MUC_MO_HINH,
    "model_config.pkl"
)


# =========================================================
# 3. TẢI MÔ HÌNH
# =========================================================

@st.cache_resource
def tai_mo_hinh():

    danh_sach_tep_can_thiet = [
        DUONG_DAN_MO_HINH_RUI_RO,
        DUONG_DAN_MO_HINH_DINH_GIA,
        DUONG_DAN_BO_HIEU_CHINH,
        DUONG_DAN_NGUONG_RUI_RO,
        DUONG_DAN_TIEN_XU_LY,
        DUONG_DAN_CAU_HINH
    ]

    tep_con_thieu = [
        tep
        for tep in danh_sach_tep_can_thiet
        if not os.path.exists(tep)
    ]

    if tep_con_thieu:
        raise FileNotFoundError(
            "Không tìm thấy các tệp mô hình sau:\n\n"
            + "\n".join(tep_con_thieu)
        )

    # -----------------------------------------------------
    # Mô hình đánh giá rủi ro
    # -----------------------------------------------------

    mo_hinh_rui_ro = CatBoostClassifier()

    mo_hinh_rui_ro.load_model(
        DUONG_DAN_MO_HINH_RUI_RO
    )

    # -----------------------------------------------------
    # Mô hình định giá
    # -----------------------------------------------------

    mo_hinh_dinh_gia = joblib.load(
        DUONG_DAN_MO_HINH_DINH_GIA
    )

    # -----------------------------------------------------
    # Bộ hiệu chỉnh xác suất
    # -----------------------------------------------------

    bo_hieu_chinh = joblib.load(
        DUONG_DAN_BO_HIEU_CHINH
    )

    # -----------------------------------------------------
    # Ngưỡng phân loại rủi ro
    # -----------------------------------------------------

    nguong_rui_ro = joblib.load(
        DUONG_DAN_NGUONG_RUI_RO
    )

    # -----------------------------------------------------
    # Bộ tiền xử lý
    # -----------------------------------------------------

    tien_xu_ly = joblib.load(
        DUONG_DAN_TIEN_XU_LY
    )

    # -----------------------------------------------------
    # Cấu hình mô hình
    # -----------------------------------------------------

    cau_hinh = joblib.load(
        DUONG_DAN_CAU_HINH
    )

    return (
        mo_hinh_rui_ro,
        mo_hinh_dinh_gia,
        bo_hieu_chinh,
        nguong_rui_ro,
        tien_xu_ly,
        cau_hinh
    )


# =========================================================
# 4. NẠP MÔ HÌNH
# =========================================================

try:

    (
        mo_hinh_rui_ro,
        mo_hinh_dinh_gia,
        bo_hieu_chinh,
        nguong_rui_ro,
        tien_xu_ly,
        cau_hinh

    ) = tai_mo_hinh()

    tai_mo_hinh_thanh_cong = True

except Exception as loi:

    tai_mo_hinh_thanh_cong = False
    thong_bao_loi = str(loi)


# =========================================================
# 5. TIÊU ĐỀ TRANG
# =========================================================

st.title(
    "🚗 Phân tích rủi ro và định giá bảo hiểm xe cơ giới"
)

st.markdown(
    """
    ### Hệ thống hỗ trợ đánh giá rủi ro và định giá bảo hiểm

    Hệ thống sử dụng học máy để:

    - Đánh giá xác suất khách hàng phát sinh yêu cầu bồi thường.
    - Phân loại mức độ rủi ro.
    - Ước lượng chi phí bồi thường kỳ vọng.
    - Đề xuất mức phí bảo hiểm tham khảo.
    """
)


# =========================================================
# 6. KIỂM TRA TRẠNG THÁI MÔ HÌNH
# =========================================================

if not tai_mo_hinh_thanh_cong:

    st.error(
        "❌ Không thể tải mô hình."
    )

    st.code(
        thong_bao_loi
    )

    st.info(
        """
        Hãy kiểm tra thư mục `models` có đầy đủ:

        - catboost_risk.cbm
        - tweedie_pricing.pkl
        - probability_calibrator.pkl
        - risk_threshold.pkl
        - preprocessor.pkl
        - model_config.pkl
        """
    )

    st.stop()

else:

    st.success(
        "✅ Hệ thống đã tải mô hình thành công."
    )


# =========================================================
# 7. THANH BÊN
# =========================================================

st.sidebar.title(
    "⚙️ Thông tin hệ thống"
)

st.sidebar.write(
    "Phân tích rủi ro và định giá bảo hiểm xe cơ giới"
)

st.sidebar.markdown(
    """
    **Mô hình đánh giá rủi ro**

    CatBoost

    **Mô hình định giá**

    Tweedie

    **Kết quả đầu ra**

    - Xác suất xảy ra bồi thường
    - Mức độ rủi ro
    - Phí thuần
    - Phí bảo hiểm đề xuất
    """
)

st.sidebar.divider()

st.sidebar.caption(
    "Hệ thống phục vụ mục đích học tập và minh họa"
)


# =========================================================
# 8. THÔNG TIN NGƯỜI LÁI
# =========================================================

st.header(
    "👤 Thông tin người lái"
)

cot1, cot2 = st.columns(2)

with cot1:

    tuoi_lai_xe = st.number_input(
        "Tuổi người lái",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

with cot2:

    bonus_malus = st.number_input(
        "Hệ số Bonus-Malus",
        min_value=50,
        max_value=200,
        value=100,
        step=1
    )


# =========================================================
# 9. THÔNG TIN PHƯƠNG TIỆN
# =========================================================

st.header(
    "🚗 Thông tin phương tiện"
)

cot3, cot4 = st.columns(2)

with cot3:

    tuoi_xe = st.number_input(
        "Tuổi của xe",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )

with cot4:

    cong_suat_xe = st.number_input(
        "Công suất xe",
        min_value=1,
        max_value=200,
        value=75,
        step=1
    )


# =========================================================
# 10. THÔNG TIN KHU VỰC
# =========================================================

st.header(
    "🌍 Thông tin khu vực"
)

cot5, cot6, cot7 = st.columns(3)

with cot5:

    khu_vuc = st.text_input(
        "Khu vực",
        value="A"
    )

with cot6:

    vung = st.text_input(
        "Vùng",
        value="R82"
    )

with cot7:

    mat_do_dan_so = st.number_input(
        "Mật độ dân số",
        min_value=1.0,
        max_value=100000.0,
        value=100.0,
        step=10.0
    )


# =========================================================
# 11. THÔNG TIN LOẠI XE
# =========================================================

st.header(
    "🚘 Thông tin loại xe"
)

cot8, cot9 = st.columns(2)

with cot8:

    thuong_hieu_xe = st.text_input(
        "Thương hiệu xe",
        value="B1"
    )

with cot9:

    loai_nhien_lieu = st.selectbox(
        "Loại nhiên liệu",
        [
            "Regular",
            "Diesel"
        ]
    )


# =========================================================
# 12. THÔNG TIN HỢP ĐỒNG
# =========================================================

st.header(
    "🛡️ Thông tin hợp đồng bảo hiểm"
)

thoi_gian_bao_hiem = st.number_input(
    "Thời gian bảo hiểm (năm)",
    min_value=0.01,
    max_value=2.0,
    value=1.0,
    step=0.01
)


# =========================================================
# 13. NÚT PHÂN TÍCH
# =========================================================

st.divider()

nut_phan_tich = st.button(
    "🔍 PHÂN TÍCH RỦI RO",
    type="primary",
    use_container_width=True
)


# =========================================================
# 14. DỰ ĐOÁN
# =========================================================

if nut_phan_tich:

    try:

        # -------------------------------------------------
        # TẠO CÁC BIẾN MỚI
        # -------------------------------------------------

        log_mat_do = np.log1p(
            mat_do_dan_so
        )

        la_nguoi_lai_tre = int(
            tuoi_lai_xe < 25
        )

        bonus_malus_cao = int(
            bonus_malus > 100
        )


        # -------------------------------------------------
        # TẠO DỮ LIỆU ĐẦU VÀO
        # -------------------------------------------------

        du_lieu_dau_vao = pd.DataFrame([{

            "VehPower": cong_suat_xe,

            "VehAge": tuoi_xe,

            "DrivAge": tuoi_lai_xe,

            "BonusMalus": bonus_malus,

            "LogDensity": log_mat_do,

            "Is_Young_Driver": la_nguoi_lai_tre,

            "High_BonusMalus": bonus_malus_cao,

            "Area": str(khu_vuc),

            "VehBrand": str(thuong_hieu_xe),

            "VehGas": str(loai_nhien_lieu),

            "Region": str(vung)

        }])


        # =================================================
        # 15. DỰ ĐOÁN XÁC SUẤT BỒI THƯỜNG
        # =================================================

        xac_suat_tho = (
            mo_hinh_rui_ro
            .predict_proba(
                du_lieu_dau_vao
            )[:, 1][0]
        )


        # -------------------------------------------------
        # Hiệu chỉnh xác suất
        # -------------------------------------------------

        xac_suat_hieu_chinh = (
            bo_hieu_chinh
            .predict_proba(
                [[xac_suat_tho]]
            )[:, 1][0]
        )


        # =================================================
        # 16. PHÂN LOẠI MỨC ĐỘ RỦI RO
        # =================================================

        if xac_suat_hieu_chinh < 0.05:

            muc_do_rui_ro = "RỦI RO THẤP"
            bieu_tuong_rui_ro = "🟢"

        elif xac_suat_hieu_chinh < 0.10:

            muc_do_rui_ro = "RỦI RO TRUNG BÌNH"
            bieu_tuong_rui_ro = "🟡"

        elif xac_suat_hieu_chinh < 0.20:

            muc_do_rui_ro = "RỦI RO CAO"
            bieu_tuong_rui_ro = "🟠"

        else:

            muc_do_rui_ro = "RỦI RO RẤT CAO"
            bieu_tuong_rui_ro = "🔴"


        # =================================================
        # 17. PHÂN LOẠI THEO NGƯỠNG
        # =================================================

        ket_qua_phan_loai = int(
            xac_suat_hieu_chinh >= nguong_rui_ro
        )

        if ket_qua_phan_loai == 1:

            nhan_xet_rui_ro = (
                "Khách hàng có khả năng phát sinh "
                "yêu cầu bồi thường cao hơn."
            )

        else:

            nhan_xet_rui_ro = (
                "Khách hàng có khả năng phát sinh "
                "yêu cầu bồi thường thấp hơn."
            )


        # =================================================
        # 18. CHUẨN BỊ DỮ LIỆU CHO ĐỊNH GIÁ
        # =================================================

        du_lieu_dinh_gia = (
            tien_xu_ly.transform(
                du_lieu_dau_vao
            )
        )


        # =================================================
        # 19. DỰ ĐOÁN PHÍ THUẦN
        # =================================================

        phi_thuan = (
            mo_hinh_dinh_gia
            .predict(
                du_lieu_dinh_gia
            )[0]
        )

        phi_thuan = max(
            float(phi_thuan),
            0.0
        )


        # =================================================
        # 20. TÍNH CHI PHÍ BỒI THƯỜNG KỲ VỌNG
        # =================================================

        chi_phi_boi_thuong_ky_vong = (
            phi_thuan *
            thoi_gian_bao_hiem
        )


        # =================================================
        # 21. TÍNH PHÍ BẢO HIỂM ĐỀ XUẤT
        # =================================================

        bien_chi_phi = float(
            cau_hinh.get(
                "expense_margin",
                0.20
            )
        )

        phi_bao_hiem_de_xuat = (
            chi_phi_boi_thuong_ky_vong
            /
            (1 - bien_chi_phi)
        )


        # =================================================
        # 22. HIỂN THỊ KẾT QUẢ
        # =================================================

        st.header(
            "📊 Kết quả phân tích"
        )


        # -------------------------------------------------
        # Các chỉ số chính
        # -------------------------------------------------

        ket_qua_1, ket_qua_2, ket_qua_3, ket_qua_4 = st.columns(4)

        with ket_qua_1:

            st.metric(
                "Xác suất xảy ra bồi thường",
                f"{xac_suat_hieu_chinh:.2%}"
            )

        with ket_qua_2:

            st.metric(
                "Mức độ rủi ro",
                f"{bieu_tuong_rui_ro} {muc_do_rui_ro}"
            )

        with ket_qua_3:

            st.metric(
                "Phí thuần",
                f"€{phi_thuan:,.2f}"
            )

        with ket_qua_4:

            st.metric(
                "Phí bảo hiểm đề xuất",
                f"€{phi_bao_hiem_de_xuat:,.2f}"
            )


        # =================================================
        # 23. BIỂU ĐỒ ĐỒNG HỒ RỦI RO
        # =================================================

        st.subheader(
            "🎯 Xác suất phát sinh bồi thường"
        )

        dong_ho = go.Figure(
            go.Indicator(

                mode="gauge+number",

                value=xac_suat_hieu_chinh * 100,

                number={
                    "suffix": "%"
                },

                title={
                    "text": "Xác suất bồi thường"
                },

                gauge={

                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": "#FF8C00"
                    },

                    "steps": [

                        {
                            "range": [0, 5],
                            "color": "#90EE90"
                        },

                        {
                            "range": [5, 10],
                            "color": "#FFD700"
                        },

                        {
                            "range": [10, 20],
                            "color": "#FFA07A"
                        },

                        {
                            "range": [20, 100],
                            "color": "#FF6B6B"
                        }

                    ]

                }

            )
        )

        dong_ho.update_layout(
            height=350
        )

        st.plotly_chart(
            dong_ho,
            use_container_width=True
        )


        # =================================================
        # 24. NHẬN XÉT NGHIỆP VỤ
        # =================================================

        st.subheader(
            "💡 Nhận xét"
        )

        st.info(
            f"""
            **Đánh giá rủi ro**

            Xác suất khách hàng phát sinh ít nhất một yêu cầu
            bồi thường được ước lượng khoảng
            **{xac_suat_hieu_chinh:.2%}**.

            Hệ thống xếp khách hàng vào nhóm:

            **{bieu_tuong_rui_ro} {muc_do_rui_ro}**

            {nhan_xet_rui_ro}

            Ngưỡng phân loại của mô hình là:
            **{nguong_rui_ro:.2%}**
            """
        )


        # =================================================
        # 25. CHI TIẾT ĐỊNH GIÁ
        # =================================================

        st.subheader(
            "💰 Chi tiết định giá"
        )

        cot_gia_1, cot_gia_2, cot_gia_3 = st.columns(3)

        with cot_gia_1:

            st.metric(
                "Phí thuần",
                f"€{phi_thuan:,.2f}"
            )

        with cot_gia_2:

            st.metric(
                "Chi phí bồi thường kỳ vọng",
                f"€{chi_phi_boi_thuong_ky_vong:,.2f}"
            )

        with cot_gia_3:

            st.metric(
                "Phí bảo hiểm đề xuất",
                f"€{phi_bao_hiem_de_xuat:,.2f}"
            )


        st.caption(
            f"""
            Biên chi phí giả định trong hệ thống:
            **{bien_chi_phi:.0%}**
            """
        )


        # =================================================
        # 26. THÔNG TIN KHÁCH HÀNG
        # =================================================

        st.subheader(
            "👤 Thông tin hồ sơ khách hàng"
        )

        bang_thong_tin = pd.DataFrame({

            "Thuộc tính": [

                "Tuổi người lái",

                "Bonus-Malus",

                "Tuổi xe",

                "Công suất xe",

                "Khu vực",

                "Vùng",

                "Thương hiệu xe",

                "Loại nhiên liệu",

                "Mật độ dân số",

                "Thời gian bảo hiểm"

            ],

            "Giá trị": [

                tuoi_lai_xe,

                bonus_malus,

                tuoi_xe,

                cong_suat_xe,

                khu_vuc,

                vung,

                thuong_hieu_xe,

                loai_nhien_lieu,

                mat_do_dan_so,

                thoi_gian_bao_hiem

            ]

        })

        st.dataframe(
            bang_thong_tin,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # 27. THÔNG TIN KỸ THUẬT
        # =================================================

        with st.expander(
            "🔎 Xem thông tin kỹ thuật"
        ):

            st.write(
                "Xác suất ban đầu:",
                f"{xac_suat_tho:.6f}"
            )

            st.write(
                "Xác suất sau hiệu chỉnh:",
                f"{xac_suat_hieu_chinh:.6f}"
            )

            st.write(
                "Ngưỡng phân loại:",
                f"{nguong_rui_ro:.6f}"
            )

            st.write(
                "Kết quả phân loại:",
                ket_qua_phan_loai
            )

            st.write(
                "Phí thuần:",
                f"€{phi_thuan:,.2f}"
            )

            st.write(
                "Chi phí bồi thường kỳ vọng:",
                f"€{chi_phi_boi_thuong_ky_vong:,.2f}"
            )

            st.write(
                "Phí bảo hiểm đề xuất:",
                f"€{phi_bao_hiem_de_xuat:,.2f}"
            )


    except Exception as loi:

        st.error(
            "❌ Không thể thực hiện dự đoán."
        )

        st.exception(loi)


# =========================================================
# 28. CHÂN TRANG
# =========================================================

st.divider()

st.caption(
    """
    Hệ thống phân tích rủi ro và định giá bảo hiểm xe cơ giới
    sử dụng Machine Learning.
    
    Lưu ý: Kết quả phí bảo hiểm là mức tham khảo phục vụ
    mục đích nghiên cứu và minh họa, không phải báo giá bảo hiểm thực tế.
    """
)