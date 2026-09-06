#include "nai_motion.h"
#include "nai_protocol.h"

#include <math.h>

namespace {

using Rep = NoodleAIModelBL702::InputRepresentation;

uint8_t channels_for(Rep rep)
{
    switch (rep) {
    case Rep::REP_ACCEL: return 3;
    case Rep::REP_GYRO: return 3;
    case Rep::REP_ACCEL_GYRO: return 6;
    case Rep::REP_QUATERNION: return 4;
    case Rep::REP_VELOCITY: return 3;
    case Rep::REP_VELOCITY_QUATERNION: return 7;
    default: return 0;
    }
}

uint8_t raw_axis(Rep rep, uint8_t c)
{
    if (rep == Rep::REP_GYRO) return c + 3u;
    return c;
}

void resample_generic(const float *src, uint16_t rawSamples,
                      uint8_t channels, float *out,
                      uint16_t normalized)
{
    if (!src || !out || !rawSamples || !normalized || !channels) return;

    if (rawSamples == 1 || normalized == 1) {
        for (uint16_t i = 0; i < normalized; ++i)
            for (uint8_t c = 0; c < channels; ++c)
                out[i * channels + c] = src[c];
        return;
    }

    const float scale =
        (float)(rawSamples - 1u) / (float)(normalized - 1u);

    for (uint16_t i = 0; i < normalized; ++i) {
        const float pos = i * scale;
        const uint16_t i0 = (uint16_t)pos;
        const uint16_t i1 = i0 + 1u < rawSamples ? i0 + 1u : i0;
        const float a = pos - (float)i0;

        for (uint8_t c = 0; c < channels; ++c) {
            const float x0 = src[i0 * channels + c];
            const float x1 = src[i1 * channels + c];
            out[i * channels + c] = x0 + a * (x1 - x0);
        }
    }
}

void resample_raw(Rep rep, const float *raw6, uint16_t rawSamples,
                  float *out, uint16_t normalized)
{
    const uint8_t channels = channels_for(rep);
    if (!raw6 || !out || !channels) return;

    if (rawSamples == 1 || normalized == 1) {
        for (uint16_t i = 0; i < normalized; ++i)
            for (uint8_t c = 0; c < channels; ++c)
                out[i * channels + c] = raw6[raw_axis(rep, c)];
        return;
    }

    const float scale =
        (float)(rawSamples - 1u) / (float)(normalized - 1u);

    for (uint16_t i = 0; i < normalized; ++i) {
        const float pos = i * scale;
        const uint16_t i0 = (uint16_t)pos;
        const uint16_t i1 = i0 + 1u < rawSamples ? i0 + 1u : i0;
        const float a = pos - (float)i0;

        for (uint8_t c = 0; c < channels; ++c) {
            const uint8_t axis = raw_axis(rep, c);
            const float x0 = raw6[i0 * 6u + axis];
            const float x1 = raw6[i1 * 6u + axis];
            out[i * channels + c] = x0 + a * (x1 - x0);
        }
    }
}

void center(float *data, uint16_t samples, uint8_t channels)
{
    float mean[7]{};
    if (!data || !samples || !channels) return;

    for (uint16_t i = 0; i < samples; ++i)
        for (uint8_t c = 0; c < channels; ++c)
            mean[c] += data[i * channels + c];

    const float inv = 1.0f / (float)samples;
    for (uint8_t c = 0; c < channels; ++c) mean[c] *= inv;

    for (uint16_t i = 0; i < samples; ++i)
        for (uint8_t c = 0; c < channels; ++c)
            data[i * channels + c] -= mean[c];
}

struct Q { float w, x, y, z; };

Q qnorm(Q q)
{
    const float n2 = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
    if (!isfinite(n2) || n2 < 1e-20f) return {1,0,0,0};
    const float inv = 1.0f / sqrtf(n2);
    q.w*=inv; q.x*=inv; q.y*=inv; q.z*=inv;
    return q;
}

Q qconj(const Q &q) { return {q.w,-q.x,-q.y,-q.z}; }

Q qmul(const Q &a, const Q &b)
{
    return {
        a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z,
        a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
        a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
        a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w
    };
}

float qdot(const Q &a, const Q &b)
{
    return a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z;
}

void qrotate(const Q &q, float x, float y, float z,
             float &ox, float &oy, float &oz)
{
    const Q p{0,x,y,z};
    const Q r = qmul(qmul(q,p),qconj(q));
    ox=r.x; oy=r.y; oz=r.z;
}

Q align_accel(float ax, float ay, float az)
{
    const float n2=ax*ax+ay*ay+az*az;
    if (!isfinite(n2) || n2<1e-12f) return {1,0,0,0};

    const float inv=1.0f/sqrtf(n2);
    const float ux=ax*inv, uy=ay*inv, uz=az*inv;
    const float d=uz;

    if (d < -0.999999f) {
        float bx = fabsf(ux)<0.9f ? 1.0f : 0.0f;
        float by = fabsf(ux)<0.9f ? 0.0f : 1.0f;
        float cx=-uz*by, cy=uz*bx, cz=ux*by-uy*bx;
        const float cn=sqrtf(cx*cx+cy*cy+cz*cz);
        if (cn<1e-8f) return {1,0,0,0};
        return {0,cx/cn,cy/cn,cz/cn};
    }
    return qnorm({1.0f+d,uy,-ux,0});
}

Q madgwick(Q q, float ax,float ay,float az,
           float gxDps,float gyDps,float gzDps,float dt)
{
    constexpr float DEG2RAD=0.01745329251994329577f;
    const float gx=gxDps*DEG2RAD, gy=gyDps*DEG2RAD, gz=gzDps*DEG2RAD;
    float q1=q.w,q2=q.x,q3=q.y,q4=q.z;

    float d1=0.5f*(-q2*gx-q3*gy-q4*gz);
    float d2=0.5f*( q1*gx+q3*gz-q4*gy);
    float d3=0.5f*( q1*gy-q2*gz+q4*gx);
    float d4=0.5f*( q1*gz+q2*gy-q3*gx);

    const float an2=ax*ax+ay*ay+az*az;
    if (isfinite(an2) && an2>1e-12f) {
        const float ia=1.0f/sqrtf(an2);
        ax*=ia; ay*=ia; az*=ia;

        const float _2q1=2*q1,_2q2=2*q2,_2q3=2*q3,_2q4=2*q4;
        const float _4q1=4*q1,_4q2=4*q2,_4q3=4*q3;
        const float _8q2=8*q2,_8q3=8*q3;
        const float q1q1=q1*q1,q2q2=q2*q2,q3q3=q3*q3,q4q4=q4*q4;

        float s1=_4q1*q3q3+_2q3*ax+_4q1*q2q2-_2q2*ay;
        float s2=_4q2*q4q4-_2q4*ax+4*q1q1*q2-_2q1*ay-_4q2+
                 _8q2*q2q2+_8q2*q3q3+_4q2*az;
        float s3=4*q1q1*q3+_2q1*ax+_4q3*q4q4-_2q4*ay-_4q3+
                 _8q3*q2q2+_8q3*q3q3+_4q3*az;
        float s4=4*q2q2*q4-_2q2*ax+4*q3q3*q4-_2q3*ay;
        const float sn2=s1*s1+s2*s2+s3*s3+s4*s4;

        if (isfinite(sn2) && sn2>1e-20f) {
            const float is=1.0f/sqrtf(sn2);
            s1*=is; s2*=is; s3*=is; s4*=is;
            d1-=NAI_MADGWICK_BETA*s1;
            d2-=NAI_MADGWICK_BETA*s2;
            d3-=NAI_MADGWICK_BETA*s3;
            d4-=NAI_MADGWICK_BETA*s4;
        }
    }

    q.w+=d1*dt; q.x+=d2*dt; q.y+=d3*dt; q.z+=d4*dt;
    return qnorm(q);
}

void store_derived(Rep rep, float *out, uint16_t k,
                   const Q &q, float vx,float vy,float vz)
{
    if (rep==Rep::REP_QUATERNION) {
        const uint16_t b=k*4u;
        out[b]=q.w; out[b+1]=q.x; out[b+2]=q.y; out[b+3]=q.z;
    } else if (rep==Rep::REP_VELOCITY) {
        const uint16_t b=k*3u;
        out[b]=vx; out[b+1]=vy; out[b+2]=vz;
    } else if (rep==Rep::REP_VELOCITY_QUATERNION) {
        const uint16_t b=k*7u;
        out[b]=vx; out[b+1]=vy; out[b+2]=vz;
        out[b+3]=q.w; out[b+4]=q.x; out[b+5]=q.y; out[b+6]=q.z;
    }
}

bool build_derived(Rep rep, const float *raw6, uint16_t n,
                   uint16_t rate, float *out)
{
    if (!raw6 || !out || n<2 || !rate) return false;

    const uint8_t channels=channels_for(rep);
    const float dt=1.0f/(float)rate;

    Q qabs=align_accel(raw6[0],raw6[1],raw6[2]);
    const Q qref=qabs;
    Q qprev{1,0,0,0};

    float awx=0,awy=0,awz=0;
    qrotate(qabs,raw6[0],raw6[1],raw6[2],awx,awy,awz);
    float axprev=awx*NAI_GRAVITY_MPS2;
    float ayprev=awy*NAI_GRAVITY_MPS2;
    float azprev=(awz-1.0f)*NAI_GRAVITY_MPS2;

    float vx=0,vy=0,vz=0;
    store_derived(rep,out,0,qprev,vx,vy,vz);

    for (uint16_t k=1;k<n;++k) {
        const uint16_t b=k*6u;
        qabs=madgwick(qabs,raw6[b],raw6[b+1],raw6[b+2],
                      raw6[b+3],raw6[b+4],raw6[b+5],dt);
        Q qrel=qnorm(qmul(qconj(qref),qabs));
        if (qdot(qrel,qprev)<0) {
            qrel.w=-qrel.w; qrel.x=-qrel.x; qrel.y=-qrel.y; qrel.z=-qrel.z;
        }
        qprev=qrel;

        qrotate(qabs,raw6[b],raw6[b+1],raw6[b+2],awx,awy,awz);
        const float axnow=awx*NAI_GRAVITY_MPS2;
        const float aynow=awy*NAI_GRAVITY_MPS2;
        const float aznow=(awz-1.0f)*NAI_GRAVITY_MPS2;

        vx += 0.5f*(axprev+axnow)*dt;
        vy += 0.5f*(ayprev+aynow)*dt;
        vz += 0.5f*(azprev+aznow)*dt;
        axprev=axnow; ayprev=aynow; azprev=aznow;

        store_derived(rep,out,k,qrel,vx,vy,vz);
    }

    if (rep==Rep::REP_VELOCITY ||
        rep==Rep::REP_VELOCITY_QUATERNION) {
        const uint16_t last=(n-1u)*channels;
        const float ex=out[last],ey=out[last+1],ez=out[last+2];

        for (uint16_t k=0;k<n;++k) {
            const float a=(float)k/(float)(n-1u);
            const uint16_t b=k*channels;
            out[b]-=a*ex; out[b+1]-=a*ey; out[b+2]-=a*ez;
        }
        out[0]=out[1]=out[2]=0;
        out[last]=out[last+1]=out[last+2]=0;
    }

    return true;
}

void normalize_quats(float *data,uint16_t samples,
                     uint8_t channels,uint8_t off)
{
    Q prev{1,0,0,0};
    for (uint16_t i=0;i<samples;++i) {
        const uint16_t b=i*channels+off;
        Q q=qnorm({data[b],data[b+1],data[b+2],data[b+3]});
        if (i && qdot(q,prev)<0) {
            q.w=-q.w;q.x=-q.x;q.y=-q.y;q.z=-q.z;
        }
        data[b]=q.w;data[b+1]=q.x;data[b+2]=q.y;data[b+3]=q.z;
        prev=q;
    }
}

} // namespace

bool nai_build_inference_representation(
    NoodleAIModelBL702::InputRepresentation rep,
    const float *raw6,
    uint16_t rawSamples,
    uint16_t sampleRateHz,
    uint16_t normalizedSamples,
    float *out,
    float *derivedScratch)
{
    const uint8_t channels=channels_for(rep);
    if (!raw6 || !rawSamples || !normalizedSamples || !out || !channels)
        return false;

    if (rep==Rep::REP_ACCEL ||
        rep==Rep::REP_GYRO ||
        rep==Rep::REP_ACCEL_GYRO) {
        resample_raw(rep,raw6,rawSamples,out,normalizedSamples);
        center(out,normalizedSamples,channels);
        return true;
    }

    if (!derivedScratch ||
        !build_derived(rep,raw6,rawSamples,sampleRateHz,derivedScratch))
        return false;

    resample_generic(derivedScratch,rawSamples,channels,out,normalizedSamples);

    if (rep==Rep::REP_QUATERNION)
        normalize_quats(out,normalizedSamples,channels,0);
    else if (rep==Rep::REP_VELOCITY_QUATERNION)
        normalize_quats(out,normalizedSamples,channels,3);

    return true;
}
