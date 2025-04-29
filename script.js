function showLiveCamera() {
    const video = document.getElementById('video');
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        video.srcObject = stream;
      })
      .catch(err => {
        alert("Could not access camera: " + err);
      });
  
    // Dummy data simulation
    setInterval(() => {
      document.getElementById('animalName').innerText = "Deer";
      document.getElementById('accuracy').innerText = Math.floor(Math.random() * 100) + "%";
    }, 2000);
  }
  
  function uploadPhoto() {
    const fileInput = document.getElementById('fileInput');
    fileInput.accept = "image/*";
    fileInput.click();
    fileInput.onchange = () => {
      alert("Image uploaded! (Backend will process this)");
    };
  }
  
  function uploadVideo() {
    const fileInput = document.getElementById('fileInput');
    fileInput.accept = "video/*";
    fileInput.click();
    fileInput.onchange = () => {
      alert("Video uploaded! (Backend will process this)");
    };
  }

  // JavaScript to toggle dark mode
// Rocker switch dark mode toggle
const modeToggle = document.getElementById('modeToggle');

// On page load, apply saved theme
if (localStorage.getItem("theme") === "dark") {
  document.body.classList.add("dark-mode");
  modeToggle.checked = true;
}

modeToggle.addEventListener('change', function () {
  if (this.checked) {
    document.body.classList.add("dark-mode");
    localStorage.setItem("theme", "dark");
  } else {
    document.body.classList.remove("dark-mode");
    localStorage.setItem("theme", "light");
  }
});
