import cv2
import base64
import uvicorn
import numpy as np
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from pydantic import BaseModel

# 설정
MODEL_NAME = "yolo11m-seg.pt"
CONF = 0.40
IOU = 0.35
IMGSZ = 960
PAD = 0.10
MAX_PAD_PX = 80

# 사람 + 동물 YOLO v11 클래스 인덱스
HUMAN_ANIMAL_IDS = {
    0,  # person
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # 동물들
    77  # teddy bear (비숑 인식용)
}
CLASS_REMAP = {77: 16}  # teddy bear -> dog

# 모델 로드 (전역)
model = None


def load_model():
    """YOLO 모델 로드"""
    global model
    if model is None:
        model = YOLO(MODEL_NAME)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 이벤트 처리"""
    # Startup
    load_model()
    print("YOLO 모델 로드 완료")
    yield
    # Shutdown (필요시 정리 작업)


def crop_with_padding(img, x1, y1, x2, y2, pad=0.10, max_pad_px=80):
    """패딩을 적용하여 이미지 크롭"""
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px = min(int(bw * pad), max_pad_px)
    py = min(int(bh * pad), max_pad_px)
    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(w, x2 + px)
    ny2 = min(h, y2 + py)
    return img[ny1:ny2, nx1:nx2], (nx1, ny1, nx2, ny2)


class CropInfo(BaseModel):
    """크롭 정보 모델"""
    global_idx: int
    crop_id: str
    src_img: str
    label: str
    conf: float
    crop_path: str
    crop_data: str  # base64 인코딩된 이미지 데이터


class DetectionResponse(BaseModel):
    """탐지 결과 응답 모델"""
    total_detections: int
    kept_detections: int
    crops: List[CropInfo]
    preview_path: str


app = FastAPI(title="YOLO11 Object Detection API", lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "YOLO11 Object Detection API", "status": "running"}


@app.post("/api/yolo/detect", response_model=DetectionResponse)
async def detect_objects_multiple(
    files: List[UploadFile] = File(...),
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    imgsz: Optional[int] = None
):
    """
    여러 이미지에서 사람과 동물을 탐지하고 크롭
    
    - **files**: 업로드할 이미지 파일들
    """
    try:
        all_crops = []
        results_list = []
        
        for img_idx, file in enumerate(files, start=1):
            # 설정값 적용
            conf_threshold = conf if conf is not None else CONF
            iou_threshold = iou if iou is not None else IOU
            img_size = imgsz if imgsz is not None else IMGSZ
            
            # 파일을 메모리에서 읽기
            content = await file.read()
            
            # numpy 배열로 변환
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                continue
            
            # 모델 로드
            yolo_model = load_model()
            
            # YOLO 추론 (numpy 배열 직접 전달)
            results = yolo_model(img, conf=conf_threshold, iou=iou_threshold, imgsz=img_size)
            r = results[0]
            names = r.names
            
            preview = img.copy()
            cls_counter = defaultdict(int)
            
            # 이미지 태그 생성
            img_tag = f"img{img_idx:02d}"
            
            kept = 0
            for b in r.boxes:
                raw_cls = int(b.cls[0])
                conf_score = float(b.conf[0])
                
                # 사람/동물만 필터링
                if raw_cls not in HUMAN_ANIMAL_IDS:
                    continue
                
                # teddy bear -> dog 매핑
                cls_id = CLASS_REMAP.get(raw_cls, raw_cls)
                label = names.get(cls_id, f"class{cls_id}")
                
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                
                # 패딩 적용하여 크롭
                crop, (nx1, ny1, nx2, ny2) = crop_with_padding(
                    img, x1, y1, x2, y2, pad=PAD, max_pad_px=MAX_PAD_PX
                )
                
                cls_counter[label] += 1
                local_idx = cls_counter[label]
                
                crop_id = f"{img_tag}_{label}_{local_idx:02d}"
                # 가상 경로 (파일 저장하지 않음)
                crop_path = f"{img_tag}/{crop_id}_conf{conf_score:.2f}.jpg"
                
                # 이미지를 base64로 인코딩
                _, buffer = cv2.imencode('.jpg', crop)
                crop_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 프리뷰에 박스 표시
                cv2.rectangle(preview, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)
                text = f"{label} {conf_score:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ty = ny1 - 6
                if ty - th - 4 < 0:
                    ty = ny1 + th + 8
                cv2.rectangle(preview, (nx1, ty - th - 4), (nx1 + tw + 4, ty), (0, 255, 0), -1)
                cv2.putText(preview, text, (nx1 + 2, ty - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                
                all_crops.append({
                    "global_idx": len(all_crops),
                    "crop_id": crop_id,
                    "src_img": file.filename,
                    "label": label,
                    "conf": conf_score,
                    "crop_path": crop_path,
                    "crop_data": crop_base64
                })
                kept += 1
            
            # 프리뷰를 base64로 인코딩
            _, preview_buffer = cv2.imencode('.jpg', preview)
            preview_base64 = base64.b64encode(preview_buffer).decode('utf-8')
            
            results_list.append({
                "img_path": file.filename,
                "total_detections": len(r.boxes),
                "kept_detections": kept,
                "preview_path": f"{img_tag}_preview.jpg",  # 가상 경로
                "preview_data": preview_base64  # base64 인코딩된 프리뷰
            })
        
        # 전체 탐지 수와 저장된 탐지 수 계산
        total_detections = sum(r["total_detections"] for r in results_list) if results_list else 0
        kept_detections = sum(r["kept_detections"] for r in results_list) if results_list else 0
        
        # 첫 번째 이미지의 프리뷰 경로 사용 (또는 마지막 이미지)
        preview_path = results_list[0]["preview_path"] if results_list else ""
        
        return {
            "total_detections": total_detections,
            "kept_detections": kept_detections,
            "crops": all_crops,
            "preview_path": preview_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")


# 파일 다운로드 엔드포인트 제거 - 모든 이미지 데이터는 base64로 응답에 포함됨


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
