#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

using namespace std;
using namespace cv;

/// FUNCTIONS ///
void fft(cv::Mat &input, cv::Mat &output);
void getIndexes(int num_channels, int num_hist_bins,
                int img_rows, int img_cols,
                int x, int y, int *index);

/// CONSTANTS ///
const double pi = 3.141592653589;

int main()
{
    // Define meta-parameters
    const bool is_single_channel = true;
    const int num_channels = 4;
    const int num_hist_bins = 256;
    const string img_locale =
            "C:\\Users\\Student\\Desktop\\Youtube\\Pics\\bilat_screenshot_04.08.2017.png";
    // "C:\\Users\\Student\\Desktop\\Personal\\Resume\\Face Pics\\DSC_0187.JPG"

    ////////////////////////////////////////
    /// Convert the image into FFT space ///
    Mat in = imread(img_locale);
    Mat out;
    if( in.empty())
    {return -1;}

    // Single channel
    if(is_single_channel == true)
    {
        cvtColor(in, in, CV_BGR2GRAY);
        fft(in, out);
    }
    // Multi-channel
    else
    {
        Mat time[3], freq[3];
        cv::split(in, time);
        for(int i = 0; i < 3; i++)
        {fft(time[i], freq[i]);}
        cv::merge(freq, 3, out);
    }

    ///////////////////////////////
    /// Acquire the audio fft's ///
    double histograms[num_channels][num_hist_bins];

    // Initialise the histograms to zero
    for(int i = 0; i < num_channels; i++)
    {
        for(int j = 0; j < num_hist_bins; j++)
        {histograms[i][j] = 0;}
    }

    // Create matrices to provide fast indexing
    Mat index_channel = Mat::zeros(in.rows, in.cols, CV_32SC1);
    Mat index_bin = Mat::zeros(in.rows, in.cols, CV_32SC1);
    for (int i = 0; i < in.rows; i++)
    {
        int* pix_chan = index_channel.ptr<int>(i);
        int* pix_bin = index_bin.ptr<int>(i);
        int index[2];
        for (int j = 0; j < in.cols; ++j)
        {
            getIndexes(num_channels, num_hist_bins, in.rows, in.cols, j, i, index);
            pix_chan[j] = index[0];
            pix_bin[j] = index[1];
        }
    }

    // Convert the FFT'd image into audio FFT's
    for (int i = 0; i < in.rows; i++)
    {
        int* pix_chan = index_channel.ptr<int>(i);
        int* pix_bin = index_bin.ptr<int>(i);
        float* pix_fft = out.ptr<float>(i);
        for (int j = 0; j < in.cols; ++j)
        {
            histograms[pix_chan[j]][pix_bin[j]] += pix_fft[j];
        }
    }

    ////////////////////////
    /// Show the results ///
    imshow("Input Image"       , in );
    imshow("spectrum Magnitude", out);

    // Print the contents of the arrays
    for(int i = 0; i < num_channels; i++)
    {
        cout << "hist" << i << " = [";
        for(int j = 0; j < num_hist_bins; j++)
        {
            if(j != 0)
            {cout << ",\t";}
            cout << histograms[i][j];
        }
        cout << "];\n" << endl;
    }

    while (true)
    {
        // Test to end the program
        if(waitKey(30) == 'q')
        {break;}
    }

    return 0;
}



void fft(cv::Mat &input, cv::Mat &output)
{
    // Courtesy of https://docs.opencv.org/3.4/d8/d01/tutorial_discrete_fourier_transform.html
    Mat I;
    input.copyTo(I);
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
    magI.copyTo(output);
}

void getIndexes(int num_channels, int num_hist_bins,
                int img_rows, int img_cols,
                int x, int y, int *index)
{
    // Initialise
    int center_x, center_y;
    center_x = img_cols/2;
    center_y = img_rows/2;

    x -= center_x;
    y -= center_y;

    // Determine which channel to select
    double theta = 0.5 + atan2(y,x) / (2*pi);
    theta *= num_channels;
    int channel = round(theta);
    index[0] = (channel == num_channels) ? 0 : channel;

    // Determine which histogram bin to select
    double max_radius = sqrt( (center_x*center_x) + (center_y*center_y) );
    double val_radius = sqrt( (x*x) + (y*y) );
    int bin = floor((val_radius / max_radius) * num_hist_bins);
    index[1] = (bin == num_hist_bins) ? num_hist_bins-1 : bin;

    return;
}

void imageToSound();

























