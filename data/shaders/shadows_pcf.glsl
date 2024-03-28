#ifdef VSG_SHADOWS_PCF
float calculateShadowCoverageForDirectionalLightPCF(inout int lightDataIndex, inout int shadowMapIndex, vec3 T, vec3 B, int extraDataSize, inout vec3 color)
{
    vec4 shadowMapSettings = lightData.values[lightDataIndex++];
    int shadowMapCount = int(shadowMapSettings.r);

    const float viableSampleRatio = 1;

    // Godot's implementation
    float diskRotation = quick_hash(gl_FragCoord.xy) * 2 * PI;
    mat2 diskRotationMatrix = mat2(cos(diskRotation), sin(diskRotation), -sin(diskRotation), cos(diskRotation));

    ivec2 sampleMask = ivec2(0, 0);
    for (int i = 0; i < POISSON_DISK_SAMPLE_COUNT; i += POISSON_DISK_SAMPLE_COUNT / min(shadowSamples, POISSON_DISK_SAMPLE_COUNT))
        sampleMask[i / 32] |= 1 << (i % 32);
    float coverage = 0;
    while (shadowMapCount > 0 && sampleMask != ivec2(0, 0))
    {
        mat4 sm_matrix = mat4(lightData.values[lightDataIndex++],
                              lightData.values[lightDataIndex++],
                              lightData.values[lightDataIndex++],
                              lightData.values[lightDataIndex++]);
        lightDataIndex += extraDataSize;

        ivec2 thisMask = ivec2(0, 0);
        for (int i = 0; i < POISSON_DISK_SAMPLE_COUNT; i += POISSON_DISK_SAMPLE_COUNT / min(shadowSamples, POISSON_DISK_SAMPLE_COUNT))
        {
            if ((sampleMask[i / 32] & 1 << (i % 32)) != 0)
            {
                float penumbraRadius = shadowMapSettings.g;
                vec2 rotatedDisk = penumbraRadius * diskRotationMatrix * POISSON_DISK[i];
                vec4 sm_tc = sm_matrix * vec4(eyePos + rotatedDisk.x * T + rotatedDisk.y * B, 1.0);
                if (sm_tc.x >= 0.0 && sm_tc.x <= 1.0 && sm_tc.y >= 0.0 && sm_tc.y <= 1.0 && sm_tc.z >= 0.0 && sm_tc.z <= 1.0)
                {
                    coverage += texture(sampler2DArrayShadow(shadowMaps, shadowMapShadowSampler), vec4(sm_tc.st, shadowMapIndex, sm_tc.z)).r;
                    thisMask[i / 32] |= 1 << (i % 32);
                }
            }
        }

        sampleMask &= ~thisMask;

#ifdef SHADOWMAP_DEBUG
        if (thisMask != ivec2(0, 0))
        {
            if (shadowMapIndex==0) color = vec3(1.0, 0.0, 0.0);
            else if (shadowMapIndex==1) color = vec3(0.0, 1.0, 0.0);
            else if (shadowMapIndex==2) color = vec3(0.0, 0.0, 1.0);
            else if (shadowMapIndex==3) color = vec3(1.0, 1.0, 0.0);
            else if (shadowMapIndex==4) color = vec3(0.0, 1.0, 1.0);
            else color = vec3(1.0, 1.0, 1.0);
        }
#endif
        ++shadowMapIndex;
        --shadowMapCount;
    }


    if (shadowMapCount > 0)
    {
        // skip lightData and shadowMap entries for shadow maps that we haven't visited for this light
        // so subsequent light positions are correct.
        lightDataIndex += (4 + extraDataSize) * shadowMapCount;
        shadowMapIndex += shadowMapCount;
    }

    return coverage / shadowSamples;
}
#endif

void skipShadowDataPCF(inout int lightDataIndex, inout int shadowMapIndex)
{
    float shadowMapCount = lightData.values[lightDataIndex++].r;
    if (shadowMapCount > 0.0)
    {
        lightDataIndex += 4 * int(shadowMapCount);
        shadowMapIndex += int(shadowMapCount);
    }
}
