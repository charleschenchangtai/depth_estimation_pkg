import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import torchvision.transforms as T
from PIL import Image as PILImage

class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')
        self.publisher_ = self.create_publisher(Image, 'depth_image', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)  # 使用USB攝像頭
        self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', force_reload=True)
        self.midas.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.midas.to(self.device)

        self.transform = T.Compose([
            T.Resize((256, 256)),  # 確保圖像尺寸為正方形，並指定高度和寬度
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture image')
            return
        
        # 將圖像從 BGR 轉換為 RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 將 numpy.ndarray 轉換為 PIL 圖像
        pil_image = PILImage.fromarray(img)  # 新增這行

        # 深度估計
        input_batch = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            #print(f"Input batch shape: {input_batch.shape}")
            prediction = self.midas(input_batch)
            #print(f"Prediction shape before interpolation: {prediction.shape}")
            # 調整插值為固定大小，例如 (256, 256)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(256, 256),  # 設定固定的輸出尺寸
                mode='bicubic',
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # 將深度圖轉換為ROS Image消息並發布
        depth_image_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding="passthrough")
        self.publisher_.publish(depth_image_msg)

        # 在窗口顯示深度圖
        cv2.imshow('Depth Map', depth_map)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
