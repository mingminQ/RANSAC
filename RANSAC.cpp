#include <vector>

cv::Mat fitParabolaRANSAC(const std::vector<cv::Point2f>& points, int maxIterations, float distanceThreshold)
{
    cv::Mat bestModel;
    int bestInliers = 0;

    // random sampling
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(0, points.size() - 1);

    
    for (int i = 0; i < maxIterations; ++i)
    {
        std::vector<cv::Point2f> samplePoints;
        for (int j = 0; j < 100; ++j) {
            int idx = dist(rng);
            samplePoints.push_back(points[idx]);
        }

        // y = ax^2 + bx + c
        cv::Mat A(3, 3, CV_32F), B(3, 1, CV_32F);
        for (int j = 0; j < 3; ++j) {
            A.at<float>(j, 0) = samplePoints[j].x * samplePoints[j].x;
            A.at<float>(j, 1) = samplePoints[j].x;
            A.at<float>(j, 2) = 1.0;
            B.at<float>(j, 0) = samplePoints[j].y;
        }

        cv::Mat model;
        cv::solve(A, B, model, cv::DECOMP_SVD);

        // 모델 평가: 인라이어 계산
        int inliers = 0;
        for (const auto& pt : points) 
        {
            float y_pred = model.at<float>(0) * pt.x * pt.x + model.at<float>(1) * pt.x + model.at<float>(2);
            if (std::abs(pt.y - y_pred) < distanceThreshold) {
                inliers++;
            }
        }

        // 인라이어가 가장 많은 모델을 선택
        if (inliers > bestInliers) 
        {
            bestInliers = inliers;
            bestModel = model;
        }
    }

    return bestModel;
}
