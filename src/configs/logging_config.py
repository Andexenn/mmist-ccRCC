import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name_log):
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Cấu trúc: [Thời gian] [Tên module] [Level] : Nội dung
    log_format = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 3. Tạo Handler ghi ra File (Lưu lại để tra cứu)
    # RotatingFileHandler: Tự động cắt file khi > 10MB, giữ lại 5 file cũ nhất
    file_handler = RotatingFileHandler(
        filename='logs/app.log', 
        maxBytes=10*1024*1024, # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.DEBUG) # File thì lưu tất cả

    # 4. Tạo Handler in ra Màn hình (Console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO) # Màn hình chỉ hiện cái quan trọng

    # 5. Khởi tạo Logger
    logger = logging.getLogger(name_log)
    logger.setLevel(logging.DEBUG)
    
    # Add handlers vào logger
    if not logger.handlers: # Tránh add trùng lặp
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger
