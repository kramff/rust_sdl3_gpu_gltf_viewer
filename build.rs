use std::process::Command;

fn main() {
    Command::new("cmd")
        .args([
            "/C",
            "glslc -fshader-stage=fragment shaders/fragment.glsl -o shaders/fragment.spv",
        ])
        .output()
        .expect("failed fragment shader");
    Command::new("cmd")
        .args([
            "/C",
            "glslc -fshader-stage=vertex shaders/vertex.glsl -o shaders/vertex.spv",
        ])
        .output()
        .expect("failed vertex shader");
}
