/**
 * File              : blend2d.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.03.2020
 * Last Modified Date: 05.03.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#include "blend2d.h"
#include "pybind/util.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

OPENPARF_BEGIN_NAMESPACE

/// A wrapper to plot
class Image {
public:
    /// @brief default constructor
    /// @param img_w image width
    /// @param img_h image height
    /// @param
    Image(std::size_t img_w, std::size_t img_h, double xl, double yl, double xh, double yh)
        : img_(img_w, img_h, BL_FORMAT_PRGB32), ctx_(img_), xl_(xl), yl_(yl), xh_(xh), yh_(yh), flipy_(true) {
        // Clear the image.
        ctx_.setCompOp(BL_COMP_OP_SRC_COPY);
        ctx_.fillAll();
        ctx_.setCompOp(BL_COMP_OP_SRC_OVER);
    }

    /// @brief set flip y flag
    void setFlipY(bool v) {
        flipy_ = v;
    }

    /// @brief fill color
    void setFillColor(uint32_t rgba) {
        ctx_.setFillStyle(BLRgba32(rgba));
    }

    /// @brief fill color
    void setFillColor(uint32_t r, uint32_t g, uint32_t b, float a) {
        ctx_.setFillStyle(BLRgba32(r, g, b, a * 255));
    }

    /// @brief stroke color
    void setStrokeColor(uint32_t rgba) {
        ctx_.setStrokeStyle(BLRgba32(rgba));
    }

    /// @brief stroke color
    void setStrokeColor(uint32_t r, uint32_t g, uint32_t b, float a) {
        ctx_.setStrokeStyle(BLRgba32(r, g, b, a * 255));
    }

    /// @brief stroke width
    void setStrokeWidth(double width) {
        ctx_.setStrokeWidth(width);
    }
    /// @brief draw a line from x1, y1 to x2, y2
    void strokeLine(double x1, double y1, double x2, double y2) {
        BLLine path(scaleX(x1), scaleY(y1), scaleX(x2), scaleY(y2));
        ctx_.strokeLine(path);
    }
    /// @brief fill a single rectangle
    void fillRect(double xl, double yl, double w, double h) {
        scaleRect(xl, yl, w, h);
        BLRect rect(xl, yl, w, h);
        ctx_.fillRectArray(&rect, 1);
    }

    /// @brief fill a set of rectangles
    /// @param pos array of (x, y) coordinates
    /// @Param sizes array of (width, height)
    void fillRects(std::vector<double> const &pos, std::vector<double> const &sizes) {
        auto rects = getRects(pos, sizes);
        ctx_.fillRectArray(rects.data(), rects.size());
    }

    /// @brief stroke a single rectangle
    void strokeRect(double xl, double yl, double w, double h) {
        scaleRect(xl, yl, w, h);
        BLRect rect(xl, yl, w, h);
        ctx_.strokeRectArray(&rect, 1);
    }

    /// @brief stroke a set of rectangles
    /// @param pos array of (x, y) coordinates
    /// @Param sizes array of (width, height)
    void strokeRects(std::vector<double> const &pos, std::vector<double> const &sizes) {
        auto rects = getRects(pos, sizes);
        ctx_.strokeRectArray(rects.data(), rects.size());
    }

    /// @brief draw text
    /// NOT well supported yet
    void text(double x, double y, std::string const &str, std::size_t size) {
        auto dst = BLPoint(scaleX(x), scaleY(y));
        auto font = BLFont();
        ctx_.fillUtf8Text(dst, font, str.c_str(), size);
    }

    /// @brief end context, must called before write
    void end() {
        ctx_.end();
    }

    /// @brief write to file
    void write(std::string const &filename) {
        img_.writeToFile(filename.c_str());
    }

protected:
    /// @brief scale the coordinates to image coordinate system
    double scaleX(double x) const {
        return (x - xl_) / (xh_ - xl_) * img_.width();
    }

    /// @brief scale the coordinates to image coordinate system
    double scaleY(double y) const {
        return (y - yl_) / (yh_ - yl_) * img_.height();
    }

    /// @brief scale width
    double scaleWidth(double w) const {
        return w / (xh_ - xl_) * img_.width();
    }

    /// @brief scale height
    double scaleHeight(double h) const {
        return h / (yh_ - yl_) * img_.height();
    }

    /// @brief scale rectangle
    void scaleRect(double &xl, double &yl, double &w, double &h) const {
        auto xxl = scaleX(xl);
        auto yyl = scaleY(yl);
        auto ww = scaleWidth(w);
        auto hh = scaleHeight(h);
        if (flipy_) {
            yyl = img_.height() - (yyl + hh);
        }
        xl = xxl;
        yl = yyl;
        w = ww;
        h = hh;
    }

    /// @brief construct array of rectangles
    std::vector<BLRect> getRects(std::vector<double> const &pos, std::vector<double> const &sizes) const {
        std::size_t n = std::min(pos.size(), sizes.size()) / 2;
        std::vector<BLRect> rects(n);
        for (std::size_t i = 0; i < n; ++i) {
            auto offset = (i << 1);
            auto xl = pos[offset];
            auto yl = pos[offset + 1];
            auto w = sizes[offset];
            auto h = sizes[offset + 1];
            scaleRect(xl, yl, w, h);
            rects[i] = BLRect(xl, yl, w, h);
        }
        return rects;
    }


    BLImage img_;             ///< image
    BLContext ctx_;           ///< render context
    double xl_, yl_, xh_, yh_;///< layout region to draw
    bool flipy_;              ///< whether flip y axis to make it consistent with coordinate system
};

OPENPARF_END_NAMESPACE

using Image = OPENPARF_NAMESPACE::Image;

void bind_blend2d(py::module &m) {
    m.doc() = R"pbdoc(
        Pybind11 plugin
        -----------------------
        .. currentmodule:: blend2d
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    py::class_<Image>(m, "Image")
            .def(py::init<std::size_t, std::size_t, double, double, double, double>())
            .def("setFlipY", (void (Image::*)(bool)) & Image::setFlipY)
            .def("setFillColor", (void (Image::*)(uint32_t)) & Image::setFillColor)
            .def("setFillColor", (void (Image::*)(uint32_t, uint32_t, uint32_t, float)) & Image::setFillColor)
            .def("setStrokeColor", (void (Image::*)(uint32_t)) & Image::setStrokeColor)
            .def("setStrokeColor", (void (Image::*)(uint32_t, uint32_t, uint32_t, float)) & Image::setStrokeColor)
            .def("setStrokeWidth", (void (Image::*)(double)) & Image::setStrokeWidth)
            .def("strokeLine", (void (Image::*)(double, double, double, double)) & Image::strokeLine)
            .def("fillRect", (void (Image::*)(double, double, double, double)) & Image::fillRect)
            .def("fillRects", [](Image &rhs,
                                 py::array_t<double, py::array::c_style | py::array::forcecast> pos,
                                 py::array_t<double, py::array::c_style | py::array::forcecast> sizes) {
                openparfAssert(pos.ndim() == 2U && pos.shape(1) == 2U);
                openparfAssert(sizes.ndim() == 2U && sizes.shape(1) == 2U);
                auto n = pos.size();
                std::vector<double> ppos;
                std::vector<double> ssizes;
                ppos.reserve(n);
                ssizes.reserve(n);
                for (std::size_t i = 0; i < (std::size_t) pos.shape(0); ++i) {
                    for (std::size_t j = 0; j < (std::size_t) pos.shape(1); ++j) {
                        ppos.push_back(pos.at(i, j));
                        ssizes.push_back(sizes.at(i, j));
                    }
                }
                rhs.fillRects(ppos, ssizes);
            })
            .def("strokeRect", (void (Image::*)(double, double, double, double)) & Image::strokeRect)
            .def("strokeRects", [](Image &rhs, py::array_t<double, py::array::c_style | py::array::forcecast> pos, py::array_t<double, py::array::c_style | py::array::forcecast> sizes) {
                openparfAssert(pos.ndim() == 2U && pos.shape(1) == 2U);
                openparfAssert(sizes.ndim() == 2U && sizes.shape(1) == 2U);
                auto n = pos.size();
                std::vector<double> ppos;
                std::vector<double> ssizes;
                ppos.reserve(n);
                ssizes.reserve(n);
                for (std::size_t i = 0; i < (std::size_t) pos.shape(0); ++i) {
                    for (std::size_t j = 0; j < (std::size_t) pos.shape(1); ++j) {
                        ppos.push_back(pos.at(i, j));
                        ssizes.push_back(sizes.at(i, j));
                    }
                }
                rhs.strokeRects(ppos, ssizes);
            })
            .def("text", (void (Image::*)(double, double, std::string const &, std::size_t)) & Image::text)
            .def("end", &Image::end)
            .def("write", (void (Image::*)(std::string const &)) & Image::write);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "develop";
#endif
}
