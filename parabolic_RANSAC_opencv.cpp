#include <vector>

cv::Mat parabolic_RANSAC(const std::vector<cv::Point2f>& points, int iteration, float inlier_distance, int sample_num)
{
    // best function varibale   
    cv::Mat bestModel;
    int bestInliers = 0;

    // generating random seed
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, points.size() - 1);

    // parabolic RANSAC fitting
    for (int repeat = 0; repeat < iteration; repeat++)
    {
        // select sample points
        std::vector<cv::Point2f> samplePoints;
        for (int sampling_iteration = 0; sampling_iteration < sample_num; sampling_iteration++) 
        {
            int sample_index = dist(rng);
            samplePoints.push_back(points[sample_index]);
        }

        // calculate parabolic function : y = a * ( x ^ 2 ) + b * x + c
        cv::Mat A(sample_num, 3, CV_32F), B(sample_num, 1, CV_32F);
        for (int idx = 0; idx < sample_num; idx++) 
        {
            // value [x]
            float* A_ptr = A.ptr<float>(idx);
            A_ptr[0] = samplePoints[idx].x * samplePoints[idx].x;
            A_ptr[1] = samplePoints[idx].x;
            A_ptr[2] = 1.0;

            // value[y]
            float* B_ptr = B.ptr<float>(idx);
            B_ptr[0] = samplePoints[idx].y;
        }

        cv::Mat model;
        cv::solve(A, B, model, cv::DECOMP_SVD);

        // evaluation inliers and model
        int inlier_count = 0;
        float* coefficient = model.ptr<float>(0);
        for (const auto& point : points)
        {
            float y_pred = coefficient[0] * point.x * point.x + coefficient[1] * point.x + coefficient[2];
            if (std::abs(point.y - y_pred) < inlier_distance) inlier_count++;
        }

        // judge the best parabolic function
        if ( bestInliers < inlier_count ) 
        {
            bestInliers = inliers;
            bestModel   = model;
        }
    }

    return bestModel;
}
