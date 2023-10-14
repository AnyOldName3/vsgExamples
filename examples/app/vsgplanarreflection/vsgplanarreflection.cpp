/* <editor-fold desc="MIT License">

Copyright(c) 2018 Robert Osfield
Copyright(c) 2020 Tim Moore
Copyright(c) 2023 Chris Djali

Portions derived from code that is Copyright (C) Sascha Willems - www.saschawillems.de

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</editor-fold> */

#include <vsg/all.h>

#ifdef vsgXchange_FOUND
#    include <vsgXchange/all.h>
#endif

#include <chrono>
#include <iostream>
#include <thread>

// Render the same scene from two viewpoints, with the primary being user-controllable,
// and the secondary being the reflection of the primary in the mirror plane.
//
// The secondary viewpoint is then used as a texture for the mirror plane so it looks
// like a real mirror. It's sampled with texelFetch as there's a texel for each screen
// pixel.
//
// TODO:
// * clip plane to exclude things on the wrong side of the mirror.
// * clip planes/scissor/stencil etc. to avoid wasting time drawing unneeded bits of
//   the secondary viewpoint.
// * culling planes to avoid wasting time telling the GPU about things only in the
//   unneeded bits of the secondary viewpoint.
// * general tidy-up.
// * view masks to disable clip planes in non-rtt view.

vsg::ref_ptr<vsg::Node> createTestScene(vsg::ref_ptr<vsg::Options> options, bool insertCullNode = true)
{
    auto builder = vsg::Builder::create();
    builder->options = options;

    auto scene = vsg::Group::create();

    vsg::GeometryInfo geomInfo;
    vsg::StateInfo stateInfo;

    geomInfo.cullNode = insertCullNode;

    // this is very carefully calibrated so nothing intersects the mirror plane as it
    // spins and I totally didn't end up with that by accident.

    geomInfo.position = geomInfo.dx * -3.0f;

    geomInfo.color.set(1.0f, 1.0f, 0.5f, 1.0f);
    scene->addChild(builder->createBox(geomInfo, stateInfo));

    geomInfo.color.set(1.0f, 0.5f, 1.0f, 1.0f);
    geomInfo.position += geomInfo.dx * 1.5f;
    scene->addChild(builder->createSphere(geomInfo, stateInfo));

    geomInfo.color.set(0.0f, 1.0f, 1.0f, 1.0f);
    geomInfo.position += geomInfo.dx * 1.5f;
    scene->addChild(builder->createCylinder(geomInfo, stateInfo));

    geomInfo.color.set(0.5f, 1.0f, 0.5f, 1.0f);
    geomInfo.position += geomInfo.dx * 1.5f;
    scene->addChild(builder->createCapsule(geomInfo, stateInfo));

    geomInfo.color.set(1.0f, 1.0f, 1.0f, 1.0f);
    geomInfo.position += geomInfo.dx * 1.5f;
    scene->addChild(builder->createBox(geomInfo, stateInfo));

    geomInfo.color.set(0.5f, 1.0f, 1.0f, 1.0f);
    geomInfo.position += geomInfo.dx * 1.5f;
    scene->addChild(builder->createSphere(geomInfo, stateInfo));

    geomInfo.color.set(1.0f, 0.0f, 1.0f, 1.0f);
    geomInfo.position += geomInfo.dx * 1.5f;
    scene->addChild(builder->createCylinder(geomInfo, stateInfo));

    geomInfo.color.set(0.5f, 0.5f, 1.0f, 1.0f);
    geomInfo.position += geomInfo.dx * 1.5f;
    scene->addChild(builder->createCapsule(geomInfo, stateInfo));

    return scene;
}

// RenderGraph for rendering to image

vsg::ref_ptr<vsg::RenderGraph> createOffscreenRendergraph(vsg::Context& context, const VkExtent2D& extent,
                                                          vsg::ImageInfo& colorImageInfo, vsg::ImageInfo& depthImageInfo)
{
    auto device = context.device;

    VkExtent3D attachmentExtent{extent.width, extent.height, 1};
    // Attachments
    // create image for color attachment
    auto colorImage = vsg::Image::create();
    colorImage->imageType = VK_IMAGE_TYPE_2D;
    colorImage->format = VK_FORMAT_R8G8B8A8_UNORM;
    colorImage->extent = attachmentExtent;
    colorImage->mipLevels = 1;
    colorImage->arrayLayers = 1;
    colorImage->samples = VK_SAMPLE_COUNT_1_BIT;
    colorImage->tiling = VK_IMAGE_TILING_OPTIMAL;
    colorImage->usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    colorImage->initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorImage->flags = 0;
    colorImage->sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto colorImageView = createImageView(context, colorImage, VK_IMAGE_ASPECT_COLOR_BIT);

    // Sampler for accessing attachment as a texture
    auto colorSampler = vsg::Sampler::create();
    colorSampler->flags = 0;
    colorSampler->magFilter = VK_FILTER_LINEAR;
    colorSampler->minFilter = VK_FILTER_LINEAR;
    colorSampler->mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    colorSampler->addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    colorSampler->addressModeV = colorSampler->addressModeU;
    colorSampler->addressModeW = colorSampler->addressModeU;
    colorSampler->mipLodBias = 0.0f;
    colorSampler->maxAnisotropy = 1.0f;
    colorSampler->minLod = 0.0f;
    colorSampler->maxLod = 1.0f;
    colorSampler->borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

    colorImageInfo.imageView = colorImageView;
    colorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorImageInfo.sampler = colorSampler;

    // create depth buffer
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    auto depthImage = vsg::Image::create();
    depthImage->imageType = VK_IMAGE_TYPE_2D;
    depthImage->extent = attachmentExtent;
    depthImage->mipLevels = 1;
    depthImage->arrayLayers = 1;
    depthImage->samples = VK_SAMPLE_COUNT_1_BIT;
    depthImage->format = depthFormat;
    depthImage->tiling = VK_IMAGE_TILING_OPTIMAL;
    depthImage->usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depthImage->initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthImage->flags = 0;
    depthImage->sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // XXX Does layout matter?
    depthImageInfo.sampler = nullptr;
    depthImageInfo.imageView = vsg::createImageView(context, depthImage, VK_IMAGE_ASPECT_DEPTH_BIT);
    depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // attachment descriptions
    vsg::RenderPass::Attachments attachments(2);
    // Color attachment
    attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // Depth attachment
    attachments[1].format = depthFormat;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    vsg::AttachmentReference colorReference = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    vsg::AttachmentReference depthReference = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    vsg::RenderPass::Subpasses subpassDescription(1);
    subpassDescription[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription[0].colorAttachments.emplace_back(colorReference);
    subpassDescription[0].depthStencilAttachments.emplace_back(depthReference);

    vsg::RenderPass::Dependencies dependencies(2);

    // XXX This dependency is copied from the offscreenrender.cpp
    // example. I don't completely understand it, but I think its
    // purpose is to create a barrier if some earlier render pass was
    // using this framebuffer's attachment as a texture.
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // This is the heart of what makes Vulkan offscreen rendering
    // work: render passes that follow are blocked from using this
    // passes' color attachment in their fragment shaders until all
    // this pass' color writes are finished.
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    auto renderPass = vsg::RenderPass::create(device, attachments, subpassDescription, dependencies);

    // Framebuffer
    auto fbuf = vsg::Framebuffer::create(renderPass, vsg::ImageViews{colorImageInfo.imageView, depthImageInfo.imageView}, extent.width, extent.height, 1);

    auto rendergraph = vsg::RenderGraph::create();
    rendergraph->renderArea.offset = VkOffset2D{0, 0};
    rendergraph->renderArea.extent = extent;
    rendergraph->framebuffer = fbuf;

    rendergraph->clearValues.resize(2);
    rendergraph->clearValues[0].color = {{0.4f, 0.2f, 0.4f, 1.0f}};
    rendergraph->clearValues[1].depthStencil = VkClearDepthStencilValue{0.0f, 0};

    return rendergraph;
}

vsg::ref_ptr<vsg::Node> createPlane(vsg::ref_ptr<vsg::ImageInfo> colorImage)
{
    auto vertexShaderSource = R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform PushConstants {
    mat4 projection;
    mat4 modelview;
} pc;

layout(location = 0) in vec3 inPosition;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = (pc.projection * pc.modelview) * vec4(inPosition, 1.0);
}
)";

    auto fragmentShaderSource = R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main() {
    int height = textureSize(texSampler, 0).y;
    outColor = texelFetch(texSampler, ivec2(gl_FragCoord.x, height - gl_FragCoord.y), 0);
}
)";

    auto vertexShader = vsg::ShaderStage::create(VK_SHADER_STAGE_VERTEX_BIT, "main", vertexShaderSource);
    auto fragmentShader = vsg::ShaderStage::create(VK_SHADER_STAGE_FRAGMENT_BIT, "main", fragmentShaderSource);

    if (!vertexShader || !fragmentShader)
    {
        std::cout << "Could not create shaders." << std::endl;
        return vsg::ref_ptr<vsg::Node>();
    }

    // set up graphics pipeline
    vsg::DescriptorSetLayoutBindings descriptorBindings{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr} // { binding, descriptorType, descriptorCount, stageFlags, pImmutableSamplers}
    };

    auto descriptorSetLayout = vsg::DescriptorSetLayout::create(descriptorBindings);

    vsg::PushConstantRanges pushConstantRanges{
        {VK_SHADER_STAGE_VERTEX_BIT, 0, 128} // projection, view, and model matrices, actual push constant calls automatically provided by the VSG's RecordTraversal
    };

    vsg::VertexInputState::Bindings vertexBindingsDescriptions{
        VkVertexInputBindingDescription{0, sizeof(vsg::vec3), VK_VERTEX_INPUT_RATE_VERTEX}, // vertex data
    };

    vsg::VertexInputState::Attributes vertexAttributeDescriptions{
        VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, // vertex data
    };

    vsg::GraphicsPipelineStates pipelineStates{
        vsg::VertexInputState::create(vertexBindingsDescriptions, vertexAttributeDescriptions),
        vsg::InputAssemblyState::create(),
        vsg::RasterizationState::create(),
        vsg::MultisampleState::create(),
        vsg::ColorBlendState::create(),
        vsg::DepthStencilState::create()};

    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{descriptorSetLayout}, pushConstantRanges);
    auto graphicsPipeline = vsg::GraphicsPipeline::create(pipelineLayout, vsg::ShaderStages{vertexShader, fragmentShader}, pipelineStates);
    auto bindGraphicsPipeline = vsg::BindGraphicsPipeline::create(graphicsPipeline);

    // create texture image and associated DescriptorSets and binding
    auto texture = vsg::DescriptorImage::create(colorImage, 0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    auto descriptorSet = vsg::DescriptorSet::create(descriptorSetLayout, vsg::Descriptors{texture});
    auto bindDescriptorSet = vsg::BindDescriptorSet::create(VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline->layout, 0, descriptorSet);

    // create StateGroup as the root of the scene/command graph to hold the GraphicsPipeline, and binding of Descriptors to decorate the whole graph
    auto scenegraph = vsg::StateGroup::create();
    scenegraph->add(bindGraphicsPipeline);
    scenegraph->add(bindDescriptorSet);

    // set up model transformation node
    auto transform = vsg::MatrixTransform::create(); // VK_SHADER_STAGE_VERTEX_BIT

    // add transform to root of the scene graph
    scenegraph->addChild(transform);

    // set up vertex and index arrays
    auto vertices = vsg::vec3Array::create(
        {{-2.0f, 5.0f, -2.0f},
         {2.0f, 5.0f, -2.0f},
         {2.0f, 5.0f, 2.0f},
         {-2.0f, 5.0f, 2.0f}}); // VK_FORMAT_R32G32B32_SFLOAT, VK_VERTEX_INPUT_RATE_INSTANCE, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE

    auto indices = vsg::ushortArray::create(
        {0, 1, 2,
         2, 3, 0}); // VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE

    // setup geometry
    auto drawCommands = vsg::Commands::create();
    drawCommands->addChild(vsg::BindVertexBuffers::create(0, vsg::DataList{vertices}));
    drawCommands->addChild(vsg::BindIndexBuffer::create(indices));
    drawCommands->addChild(vsg::DrawIndexed::create(6, 1, 0, 0, 0));

    drawCommands->setValue("IsReflectionPlane", true);

    // add drawCommands to transform
    transform->addChild(drawCommands);
    return scenegraph;
}

vsg::ref_ptr<vsg::Camera> createCameraForScene(vsg::Node* scenegraph, const VkExtent2D& extent)
{
    // compute the bounds of the scene graph to help position camera
    vsg::ComputeBounds computeBounds;
    scenegraph->accept(computeBounds);
    vsg::dvec3 centre = (computeBounds.bounds.min + computeBounds.bounds.max) * 0.5;
    double radius = vsg::length(computeBounds.bounds.max - computeBounds.bounds.min) * 0.6;
    double nearFarRatio = 0.001;

    // set up the camera
    auto lookAt = vsg::LookAt::create(centre + vsg::dvec3(0.0, -radius * 3.5, 0.0),
                                      centre, vsg::dvec3(0.0, 0.0, 1.0));

    auto perspective = vsg::Perspective::create(30.0, static_cast<double>(extent.width) / static_cast<double>(extent.height),
                                                nearFarRatio * radius, radius * 4.5 * 2);

    return vsg::Camera::create(perspective, lookAt, vsg::ViewportState::create(extent));
}

class FindPlanes : public vsg::Inherit<vsg::Visitor, FindPlanes>
{
public:
    std::map<vsg::RefObjectPath, vsg::ref_ptr<vsg::Object>> planes;

    void apply(vsg::Object& object) override
    {
        _objectPath.push_back(&object);

        bool isReflectionPlane = false;
        if (object.getValue("IsReflectionPlane", isReflectionPlane) && isReflectionPlane)
        {
            vsg::RefObjectPath convertedPath(_objectPath.begin(), _objectPath.end());
            planes[convertedPath] = &object;
        }

        object.traverse(*this);

        _objectPath.pop_back();
    }

protected:
    vsg::ObjectPath _objectPath;
};

class PreMultipliedRelativeViewMatrix : public vsg::Inherit<vsg::ViewMatrix, PreMultipliedRelativeViewMatrix>
{
public:
    PreMultipliedRelativeViewMatrix(const vsg::dmat4& m, vsg::ref_ptr<vsg::ViewMatrix> vm):
        matrix(m),
        viewMatrix(vm)
    {
    }

    /// returns matrix * viewMatrix->transform()
    vsg::dmat4 transform() const override
    {
        vsg::dmat4 reflectY(1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);
        return reflectY * viewMatrix->transform() * matrix;
    }

    vsg::dmat4 matrix;
    vsg::ref_ptr<vsg::ViewMatrix> viewMatrix;
};

// I don't know the VSG policy on the terms master and slave
// followerCamera must already have PreMultipliedRelativeViewMatrix
void reflectCameraPose(PreMultipliedRelativeViewMatrix& followerCameraViewMatrix, vsg::Camera& leaderCamera, const vsg::RefObjectPath& planePath)
{
    vsg::plane reflectionPlane(vsg::vec3(0.0f, 5.0f, 0.0f), vsg::vec3(0.0f, -1.0f, 0.0f));
    auto inverseTransform = vsg::inverse(vsg::computeTransform(planePath));
    reflectionPlane = reflectionPlane * static_cast<vsg::mat4>(inverseTransform);

    const auto p = static_cast<vsg::dvec3>(reflectionPlane.n) * static_cast<double>(-reflectionPlane.p);
    const auto n = static_cast<vsg::dvec3>(reflectionPlane.n);
    const auto pon = vsg::dot(p, n);


    followerCameraViewMatrix.matrix = {
        1 - 2 * n.x * n.x, -2 * n.x * n.y, -2 * n.x * n.z, 2 * pon * n.x,
        -2 * n.x * n.y, 1 - 2 * n.y * n.y, -2 * n.y * n.z, 2 * pon * n.y,
        -2 * n.x * n.z, -2 * n.y * n.z, 1 - 2 * n.z * n.z, 2 * pon * n.z,
        0, 0, 0, 1
    };
    followerCameraViewMatrix.matrix = vsg::transpose(followerCameraViewMatrix.matrix);
    
    followerCameraViewMatrix.viewMatrix = leaderCamera.viewMatrix;
}

int main(int argc, char** argv)
{
    // set up defaults and read command line arguments to override them
    vsg::CommandLine arguments(&argc, argv);

    auto windowTraits = vsg::WindowTraits::create();
    windowTraits->windowTitle = "planarreflection";
    windowTraits->debugLayer = arguments.read({"--debug", "-d"});
    windowTraits->apiDumpLayer = arguments.read({"--api", "-a"});
    windowTraits->synchronizationLayer = arguments.read("--sync");
    if (arguments.read({"--window", "-w"}, windowTraits->width, windowTraits->height)) { windowTraits->fullscreen = false; }

    bool nestedCommandGraph = arguments.read({"-n", "--nested"});
    bool separateCommandGraph = arguments.read("-s");
    bool multiThreading = arguments.read("--mt");

    bool insertCullNode = arguments.read("--cull");

    if (arguments.errors()) return arguments.writeErrorMessages(std::cerr);

    // read shaders
    vsg::Paths searchPaths = vsg::getEnvPaths("VSG_FILE_PATH");

    using VsgNodes = std::vector<vsg::ref_ptr<vsg::Node>>;
    VsgNodes vsgNodes;

    auto options = vsg::Options::create();
    options->fileCache = vsg::getEnv("VSG_FILE_CACHE");
    options->paths = vsg::getEnvPaths("VSG_FILE_PATH");

#ifdef vsgXchange_all
    // add vsgXchange's support for reading and writing 3rd party file formats
    options->add(vsgXchange::all::create());
#endif

    // read any vsg files
    for (int i = 1; i < argc; ++i)
    {
        vsg::Path filename = arguments[i];
        auto loaded_scene = vsg::read_cast<vsg::Node>(filename, options);
        if (loaded_scene)
        {
            vsgNodes.push_back(loaded_scene);
            arguments.remove(i, 1);
            --i;
        }
    }

    // assign the vsg_scene from the loaded nodes
    vsg::ref_ptr<vsg::Node> vsg_scene;
    if (vsgNodes.size() > 1)
    {
        auto vsg_group = vsg::Group::create();
        for (auto& subgraphs : vsgNodes)
        {
            vsg_group->addChild(subgraphs);
        }

        vsg_scene = vsg_group;
    }
    else if (vsgNodes.size() == 1)
    {
        vsg_scene = vsgNodes.front();
    }
    else
    {
        vsg_scene = createTestScene(options, insertCullNode);
    }

    // A hack for getting the example teapot into the correct orientation
    auto zUp = vsg::MatrixTransform::create(vsg::dmat4(1.0, 0.0, 0.0, 0.0,
                                                       0.0, 0.0, -1.0, 0.0,
                                                       0.0, 1.0, 0.0, 0.0,
                                                       0.0, 0.0, 0.0, 1.0));
    zUp->addChild(vsg_scene);

    // Transform for rotation animation
    auto transform = vsg::MatrixTransform::create();
    transform->addChild(zUp);
    vsg_scene = transform;

    // create the viewer and assign window(s) to it
    auto viewer = vsg::Viewer::create();
    auto window = vsg::Window::create(windowTraits);
    if (!window)
    {
        std::cout << "Could not create window." << std::endl;
        return 1;
    }

    viewer->addWindow(window);

    // Add explicit light so it's shared between views and has a consistent direction for both.
    vsg::ref_ptr<vsg::Group> lightGroup = vsg::Group::create();

    auto ambientLight = vsg::AmbientLight::create();
    ambientLight->name = "ambient";
    ambientLight->color.set(1.0f, 1.0f, 1.0f);
    ambientLight->intensity = 0.05f;
    lightGroup->addChild(ambientLight);

    auto directionalLight = vsg::DirectionalLight::create();
    directionalLight->name = "sunlight";
    directionalLight->color.set(1.0f, 1.0f, 1.0f);
    directionalLight->intensity = 0.95f;
    directionalLight->direction.set(0.2f, 0.2f, -0.717157f);
    lightGroup->addChild(directionalLight);

    lightGroup->addChild(vsg_scene);
    vsg_scene = lightGroup;

    auto context = vsg::Context::create(window->getOrCreateDevice());

    // Framebuffer with attachments
    auto offscreenCamera = createCameraForScene(vsg_scene, window->extent2D());
    auto offscreenCameraViewMatrix = PreMultipliedRelativeViewMatrix::create(vsg::dmat4(), offscreenCamera->viewMatrix);
    offscreenCamera->viewMatrix = offscreenCameraViewMatrix;
    auto colorImage = vsg::ImageInfo::create();
    auto depthImage = vsg::ImageInfo::create();
    auto rtt_RenderGraph = createOffscreenRendergraph(*context, window->extent2D(), *colorImage, *depthImage);
    auto rtt_view = vsg::View::create(offscreenCamera, vsg_scene);
    rtt_RenderGraph->addChild(rtt_view);

    // Plane geometry that uses the rendered scene as a texture map
    vsg::ref_ptr<vsg::Node> plane = createPlane(colorImage);
    vsg::ref_ptr<vsg::Group> group = vsg::Group::create();
    group->addChild(plane);
    group->addChild(vsg_scene);
    vsg_scene = group;

    auto camera = createCameraForScene(vsg_scene, window->extent2D());
    offscreenCamera->projectionMatrix = camera->projectionMatrix;
    auto main_RenderGraph = vsg::createRenderGraphForView(window, camera, vsg_scene, VK_SUBPASS_CONTENTS_INLINE, false);
    // TODO: set masks on view, which is child of rendergraph

    // add close handler to respond to the close window button and pressing escape
    viewer->addEventHandler(vsg::CloseHandler::create(viewer));

    viewer->addEventHandler(vsg::Trackball::create(camera));

    if (nestedCommandGraph)
    {
        std::cout << "Nested CommandGraph, with nested RTT CommandGraph as a child on the main CommandGraph. " << std::endl;
        auto rtt_commandGraph = vsg::CommandGraph::create(window);
        rtt_commandGraph->submitOrder = -1; // render before the main_commandGraph
        rtt_commandGraph->addChild(rtt_RenderGraph);

        auto main_commandGraph = vsg::CommandGraph::create(window);
        main_commandGraph->addChild(main_RenderGraph);
        main_commandGraph->addChild(rtt_commandGraph); // rtt_commandGraph nested within main CommandGraph

        viewer->assignRecordAndSubmitTaskAndPresentation({main_commandGraph});
    }
    else if (separateCommandGraph)
    {
        std::cout << "Seperate CommandGraph with RTT CommandGraph first, then main CommandGraph second." << std::endl;
        auto rtt_commandGraph = vsg::CommandGraph::create(window);
        rtt_commandGraph->submitOrder = -1; // render before the main_commandGraph
        rtt_commandGraph->addChild(rtt_RenderGraph);

        auto main_commandGraph = vsg::CommandGraph::create(window);
        main_commandGraph->addChild(main_RenderGraph);

        viewer->assignRecordAndSubmitTaskAndPresentation({rtt_commandGraph, main_commandGraph});
    }
    else
    {
        std::cout << "Single CommandGraph containing by the RTT and main RenderGraphs" << std::endl;
        // Place the offscreen RenderGraph before the plane geometry RenderGraph
        auto commandGraph = vsg::CommandGraph::create(window);
        commandGraph->addChild(rtt_RenderGraph);
        commandGraph->addChild(main_RenderGraph);

        viewer->assignRecordAndSubmitTaskAndPresentation({commandGraph});
    }

    viewer->compile();

    if (multiThreading)
    {
        std::cout << "Enabled multi-threading" << std::endl;
        viewer->setupThreading();
    }

    // rendering main loop
    while (viewer->advanceToNextFrame())
    {
        // pass any events into EventHandlers assigned to the Viewer
        viewer->handleEvents();

        // animate the offscreen scenegraph
        float time = std::chrono::duration<float, std::chrono::seconds::period>(viewer->getFrameStamp()->time - viewer->start_point()).count();
        transform->matrix = vsg::rotate(time * vsg::radians(90.0f), vsg::vec3(0.0f, 0.0, 1.0f));

        FindPlanes findPlanes;
        vsg_scene->accept(findPlanes);
        for (const auto& [objectPath, plane] : findPlanes.planes)
            reflectCameraPose(*offscreenCameraViewMatrix, *camera, objectPath);

        viewer->update();

        viewer->recordAndSubmit();

        viewer->present();
    }

    // clean up done automatically thanks to ref_ptr<>
    return 0;
}
