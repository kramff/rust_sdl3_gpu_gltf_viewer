use std::process::Command;

fn main() {
    let vertex_output = Command::new("cmd")
        .args([
            "/C",
            // "glslc -fshader-stage=vertex shaders/vertex.glsl -o shaders/vertex.spv",
            "glslc -fshader-stage=vertex shaders/vertex.glsl -o shaders/vertex.spv -g",
        ])
        .output()
        // .expect("failed vertex shader");
        .unwrap();
    if vertex_output.stderr.len() > 0 {
        println!(
            "cargo::error=Output from vertex shader compile: {:?}",
            String::from_utf8(vertex_output.stderr)
                .unwrap_or(String::from("Issue in vertex shader?"))
        );
    }
    let fragment_output = Command::new("cmd")
        .args([
            "/C",
            // "glslc -fshader-stage=fragment shaders/fragment.glsl -o shaders/fragment.spv",
            "glslc -fshader-stage=fragment shaders/fragment.glsl -o shaders/fragment.spv -g",
        ])
        .output()
        // .expect("failed fragment shader");
        .unwrap();
    if fragment_output.stderr.len() > 0 {
        println!(
            "cargo::error=Output from fragment shader compile: {:?}",
            String::from_utf8(fragment_output.stderr)
                .unwrap_or(String::from("Issue in fragment shader?"))
        );
    }
}
