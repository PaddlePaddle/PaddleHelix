#ifndef LINEAR_RNA_UTILS_FAST_MATH_H
#define LINEAR_RNA_UTILS_FAST_MATH_H

#include <math.h> 
#include <assert.h>

#define kT 61.63207755
#define NEG_INF -2e20
#define VALUE_MIN std::numeric_limits<double>::lowest()

// log space: borrowed from CONTRAfold
inline float fast_log_exp_plus_one(float x){  
    assert(float(0.0000000000) <= x && x <= float(11.8624794162) && "Argument out-of-range.");
    if (x < float(3.3792499610))
    {
        if (x < float(1.6320158198))
        {
            if (x < float(0.6615367791))
                return ((float(-0.0065591595)*x+float(0.1276442762))*x+float(0.4996554598))*x+float(0.6931542306);
            return ((float(-0.0155157557)*x+float(0.1446775699))*x+float(0.4882939746))*x+float(0.6958092989);
        }
        if (x < float(2.4912588184))
            return ((float(-0.0128909247)*x+float(0.1301028251))*x+float(0.5150398748))*x+float(0.6795585882);
        return ((float(-0.0072142647)*x+float(0.0877540853))*x+float(0.6208708362))*x+float(0.5909675829);
    }
    if (x < float(5.7890710412))
    {
        if (x < float(4.4261691294))
            return ((float(-0.0031455354)*x+float(0.0467229449))*x+float(0.7592532310))*x+float(0.4348794399);
        return ((float(-0.0010110698)*x+float(0.0185943421))*x+float(0.8831730747))*x+float(0.2523695427);
    }
    if (x < float(7.8162726752))
        return ((float(-0.0001962780)*x+float(0.0046084408))*x+float(0.9634431978))*x+float(0.0983148903);
    return ((float(-0.0000113994)*x+float(0.0003734731))*x+float(0.9959107193))*x+float(0.0149855051);
}

inline void fast_log_plus_equals (float &x, float y)
{
    if (x < y) std::swap (x, y);
    if (y > float(NEG_INF/2) && x-y < float(11.8624794162))
        x = fast_log_exp_plus_one(x-y) + y;
}

inline float fast_exp(float x)
{    
    if (x < float(-2.4915033807))
    {
        if (x < float(-5.8622823336))
        {
            if (x < float(-9.91152))
                return float(0);
            return ((float(0.0000803850)*x+float(0.0021627428))*x+float(0.0194708555))*x+float(0.0588080014);
        }
        if (x < float(-3.8396630909))
            return ((float(0.0013889414)*x+float(0.0244676474))*x+float(0.1471290604))*x+float(0.3042757740);
        return ((float(0.0072335607)*x+float(0.0906002677))*x+float(0.3983111356))*x+float(0.6245959221);
    }
    if (x < float(-0.6725053211))
    {
        if (x < float(-1.4805375919))
            return ((float(0.0232410351)*x+float(0.2085645908))*x+float(0.6906367911))*x+float(0.8682322329);
        return ((float(0.0573782771)*x+float(0.3580258429))*x+float(0.9121133217))*x+float(0.9793091728);
    }
    if (x < float(0))
        return ((float(0.1199175927)*x+float(0.4815668234))*x+float(0.9975991939))*x+float(0.9999505077);
    return (x > float(46.052) ? float(1e20) : expf(x));
}

#endif // LINEAR_RNA_UTILS_FAST_MATH_H
