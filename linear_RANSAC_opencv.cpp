#include <vector>

cv::Mat linear_RANSAC(const std::vector<cv::Point2f>& points, int iteration, double inlier_distance, int sample_num)
{
    // best function varibale   
    cv::Mat bestModel;
    int bestInliers = 0;
    
    // random sampling
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, points.size() - 1);
    
    // linear RANSAC fitting
    for (int repeat = 0; repeat < iteration; repeat++)
    {
        int sample_idx1 = dist(rng); cv::Point2f point1 = points[sample_idx1];
        int sample_idx2 = dist(rng); cv::Point2f point2 = points[sample_idx2];

        // calculate linear function : ax + by + c = 0
        double a = point2.y - point1.y;
        double b = point1.x - point2.x;
        double c = point2.x * point1.y - point1.x * point2.y;

        // calculate inliers based on distance
        int inlier_count = 0;
        for (const auto& point : points)
        {
            double distance = std::abs(a * point.x + b * point.y + c) / std::sqrt(a * a + b * b);
            if (distance < inlier_distance) inlier_count++;
        }

        // select the model that has the best inliers
        if (bestInliers < inlier_count)
        {
            bestInliers = inlier_count;
            best_model = (cv::Mat_<float>(1, 3) << a, b, c);
        }
    }

    // no model found
    if (best_model.empty()) return cv::Mat();

    // transform ax + by + c = 0 -> y = a'x + b'
    float a = best_model.at<float>(0, 0);
    float b = best_model.at<float>(0, 1);
    float c = best_model.at<float>(0, 2);

    // return slope and intercept
    float slope     = -a / b;
    float intercept = -c / b;

    return (cv::Mat_<float>(1, 2) << slope, intercept);
}
