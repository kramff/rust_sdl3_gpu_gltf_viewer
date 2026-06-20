use std::process::Command;

fn main() {
    // "glslc -fshader-stage=vertex shaders/vertex.glsl -o shaders/vertex.spv",
    let vertex_shader_compile_command =
        "glslc -fshader-stage=vertex shaders/vertex.glsl -o shaders/vertex.spv -g";
    let vertex_output = if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(["/C", vertex_shader_compile_command])
            .output()
            .unwrap()
    } else {
        Command::new("sh")
            .arg("-c")
            .arg(vertex_shader_compile_command)
            .output()
            .unwrap()
    };
    if vertex_output.stderr.len() > 0 {
        println!(
            "cargo::error=Output from vertex shader compile: {:?}",
            String::from_utf8(vertex_output.stderr)
                .unwrap_or(String::from("Issue in vertex shader?"))
        );
    }
    // "glslc -fshader-stage=fragment shaders/fragment.glsl -o shaders/fragment.spv",
    let fragment_shader_compile_command =
        "glslc -fshader-stage=fragment shaders/fragment.glsl -o shaders/fragment.spv -g";
    let fragment_output = if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(["/C", fragment_shader_compile_command])
            .output()
            .unwrap()
    } else {
        Command::new("sh")
            .arg("-c")
            .arg(fragment_shader_compile_command)
            .output()
            .unwrap()
    };
    if fragment_output.stderr.len() > 0 {
        println!(
            "cargo::error=Output from fragment shader compile: {:?}",
            String::from_utf8(fragment_output.stderr)
                .unwrap_or(String::from("Issue in fragment shader?"))
        );
    }
}
