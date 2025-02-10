    // clang-format off
  private:
    real_matrix stagewise_storage = [this] {
        auto [N, nx, nu, ny, ny_N] = dim;
        return real_matrix{{
            .depth = N + 1,
            .rows = nx*(3*nx) + nx*(nu + nx) + nx*(nu + nx) + 4*nx*1 + ny*(nu + nx) + 2*ny*1 + (nu + nx)*(nu + 2*nx) + (nu + nx)*(nu + nx) + 5*(nu + nx)*1,
            .cols = 1,
        }};
    }();


  public:
    const mut_real_view AB = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(0, nx*(nu + nx)).reshaped(nx, nu + nx);
    }();

  public:
    const mut_real_view CD = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(nu*nx + nx*nx, ny*(nu + nx)).reshaped(ny, nu + nx);
    }();

  public:
    const mut_real_view H = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(nu*nx + nu*ny + nx*ny + nx*nx, (nu + nx)*(nu + nx)).reshaped(nu + nx, nu + nx);
    }();

  public:
    const mut_real_view LHV = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(3*nu*nx + nu*ny + nu*nu + nx*ny + 2*(nx*nx), (nu + nx)*(nu + 2*nx)).reshaped(nu + 2*nx, nu + nx);
    }();

  public:
    const mut_real_view Wᵀ = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(6*nu*nx + nu*ny + 2*(nu*nu) + nx*ny + 4*(nx*nx), nx*(nu + nx)).reshaped(nu + nx, nx);
    }();

  public:
    const mut_real_view LΨU = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(7*nu*nx + nu*ny + 2*(nu*nu) + nx*ny + 5*(nx*nx), nx*(3*nx)).reshaped(3*nx, nx);
    }();

  public:
    const mut_real_view Σb = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(7*nu*nx + nu*ny + 2*(nu*nu) + nx*ny + 8*(nx*nx), ny*1).reshaped(ny, 1);
    }();

  public:
    const mut_real_view xb = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(7*nu*nx + nu*ny + 2*(nu*nu) + nx*ny + 8*(nx*nx) + ny, (nu + nx)*1).reshaped(nu + nx, 1);
    }();

  public:
    const mut_real_view bb = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(7*nu*nx + nu*ny + nu + 2*(nu*nu) + nx*ny + nx + 8*(nx*nx) + ny, nx*1).reshaped(nx, 1);
    }();

  public:
    const mut_real_view Mxbb = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(7*nu*nx + nu*ny + nu + 2*(nu*nu) + nx*ny + 2*nx + 8*(nx*nx) + ny, nx*1).reshaped(nx, 1);
    }();

  public:
    const mut_real_view db = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(7*nu*nx + nu*ny + nu + 2*(nu*nu) + nx*ny + 3*nx + 8*(nx*nx) + ny, (nu + nx)*1).reshaped(nu + nx, 1);
    }();

  public:
    const mut_real_view Δλb = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(7*nu*nx + nu*ny + 2*nu + 2*(nu*nu) + nx*ny + 4*nx + 8*(nx*nx) + ny, nx*1).reshaped(nx, 1);
    }();

  public:
    const mut_real_view Aᵀŷb = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(7*nu*nx + nu*ny + 2*nu + 2*(nu*nu) + nx*ny + 5*nx + 8*(nx*nx) + ny, (nu + nx)*1).reshaped(nu + nx, 1);
    }();

  public:
    const mut_real_view MᵀΔλb = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(7*nu*nx + nu*ny + 2*(nu*nu) + 3*nu + nx*ny + 6*nx + 8*(nx*nx) + ny, (nu + nx)*1).reshaped(nu + nx, 1);
    }();

  public:
    const mut_real_view Mᵀλb = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(7*nu*nx + nu*ny + 2*(nu*nu) + 4*nu + nx*ny + 7*nx + 8*(nx*nx) + ny, (nu + nx)*1).reshaped(nu + nx, 1);
    }();

  public:
    const mut_real_view λb = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(7*nu*nx + nu*ny + 2*(nu*nu) + 5*nu + nx*ny + 8*nx + 8*(nx*nx) + ny, nx*1).reshaped(nx, 1);
    }();

  public:
    const mut_real_view ŷb = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(7*nu*nx + nu*ny + 2*(nu*nu) + 5*nu + nx*ny + 8*(nx*nx) + 9*nx + ny, ny*1).reshaped(ny, 1);
    }();

    // clang-format on
