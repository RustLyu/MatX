Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

AOCL-BLAS 5.0.0 on Windows - version
-------------------------------------------------------------------------------
PROCEDURE TO USE SINGLE THREAD DLL IN RUN-TIME DYNAMIC LINKING TO TEST DGEMM API:
1.) Create a Visual Studio console project with the provided example_dgemm_st.c
    file inside examples folder.

2.) Add AOCL-BLAS dll path to LoadLibrary() in the source file example_dgemm_st.c.
    Eg: LoadLibrary(TEXT("..\\lib\\AOCL-LibBlis-Win-dll.dll"));

3.) Compile the application and run.


FOLLOW BELOW PROCEDURE TO TEST ANY OTHER API (EG: DTRSM, SGEMM, etc):
1.) Declare a function pointer with appropriate API signature in
    example_<API-Name>_st.c
    Eg: typedef <Return type> (*Fptr_API)(...API parameters...);

2.) Pass the API to GetProcAddress(hModule, "API") in the source file
    example_<API-Name>_st.c
    Eg: GetProcAddress(hModule1, "<API-Name>");

3.) GetProcAddress returns a handle for <API-Name>. Call the API with the handle.

4.) Unload the library by calling FreeLibrary(handle).


PROCEDURE TO USE MULTI THREAD DLL IN RUN-TIME DYNAMIC LINKING:
1.) Use example_dgemm_mt.c to test multi thread dll and follow the steps similar
    to single thread dll explained above.

2.) AOCL-BLAS requires LIBOMP library for multithreading. So, Load libomp.dll as the
    first handle and then load AOCL-BLAS dll as the second handle.
example_dgemm_mt.c can be used for this purpose.

-------------------------------------------------------------------------------
Note: The library path should be a relative path of the library from the output
      directory where the .exe(app) is getting generated.
-------------------------------------------------------------------------------
